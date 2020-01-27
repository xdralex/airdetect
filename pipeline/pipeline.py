import click
import logging
import math
from typing import Dict, NamedTuple, List, Union

import hyperopt
import numpy as np
import pandas as pd
import torch
import yaml
from hyperopt import hp, fmin
from tabulate import tabulate
from tensorboard import program
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from wheel5 import logutils
from wheel5.dataset import LMDBImageDataset, WrappingTransformDataset, split_indices
from wheel5.model import fit, score
from wheel5.tracking import Tracker, Snapshotter, CheckpointSnapshotter, BestCVSnapshotter, SnapshotConfig, TensorboardConfig
from wheel5.transforms import SquarePaddedResize

from data import load_aircraft_data


class DataBundle(NamedTuple):
    loader: DataLoader
    dataset: Dataset
    indices: List[int]


class PipelineData(NamedTuple):
    train: DataBundle
    public_test: DataBundle
    private_test: DataBundle


def launch_tensorboard(tensorboard_root: str, port: int = 6006):
    logger = logging.getLogger(PIPELINE_LOGGER)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--bind_all', '--port', f'{port}', '--logdir', tensorboard_root])
    url = tb.launch()

    logger.info(f'Launched TensorBoard at {url}')


def load_image_dataset(dataset_config: Dict[str, str], db_config: Dict[str, str]) -> LMDBImageDataset:
    df_images = load_aircraft_data(db_config, 'aircraft_photos_snapshot')

    return LMDBImageDataset.cached(df_images,
                                   image_dir=dataset_config['image_dir'],
                                   lmdb_path=dataset_config['lmdb_dir'],
                                   prepare_transform=SquarePaddedResize(size=224))


def wrap_model_dataset(dataset: Dataset) -> WrappingTransformDataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def transform_fn(arg):
        image, cls, name, index = arg
        return transform(image), cls, name, index

    return WrappingTransformDataset(
        wrapped=dataset,
        transform_fn=transform_fn
    )


# FIXME: train and test should be stored and loaded separately from the very beginning
def load_data(datasets_config: Dict, db_config: Dict, grad_batch: int = 64, nograd_batch: int = 256) -> PipelineData:
    image_dataset = load_image_dataset(datasets_config['original'], db_config)
    model_dataset = wrap_model_dataset(image_dataset)

    random_state = np.random.RandomState(42)
    indices = list(range(len(model_dataset)))

    test_indices, train_indices = split_indices(indices, split=0.25, random_state=random_state)
    public_test_indices, private_test_indices = split_indices(test_indices, split=0.5, random_state=random_state)

    train_sampler = SubsetRandomSampler(train_indices)
    public_test_sampler = SubsetRandomSampler(public_test_indices)
    private_test_sampler = SubsetRandomSampler(private_test_indices)

    train_loader = DataLoader(model_dataset, batch_size=grad_batch, sampler=train_sampler, num_workers=4, pin_memory=True)
    public_test_loader = DataLoader(model_dataset, batch_size=nograd_batch, sampler=public_test_sampler, num_workers=4, pin_memory=True)
    private_test_loader = DataLoader(model_dataset, batch_size=nograd_batch, sampler=private_test_sampler, num_workers=4, pin_memory=True)

    return PipelineData(train=DataBundle(loader=train_loader, dataset=model_dataset, indices=train_indices),
                        public_test=DataBundle(loader=public_test_loader, dataset=model_dataset, indices=public_test_indices),
                        private_test=DataBundle(loader=private_test_loader, dataset=model_dataset, indices=private_test_indices))


def split_eval_main_data(bundle: DataBundle, split: float, grad_batch: int = 64, nograd_batch: int = 256) -> (DataBundle, DataBundle):
    random_state = np.random.RandomState(42)

    eval_indices, main_indices = split_indices(bundle.indices, split=split, random_state=random_state)

    eval_sampler = SubsetRandomSampler(eval_indices)
    main_sampler = SubsetRandomSampler(main_indices)

    eval_loader = DataLoader(bundle.dataset, batch_size=nograd_batch, sampler=eval_sampler, num_workers=4, pin_memory=True)
    main_loader = DataLoader(bundle.dataset, batch_size=grad_batch, sampler=main_sampler, num_workers=4, pin_memory=True)

    eval_bundle = DataBundle(loader=eval_loader, dataset=bundle.dataset, indices=eval_indices)
    main_bundle = DataBundle(loader=main_loader, dataset=bundle.dataset, indices=main_indices)

    return eval_bundle, main_bundle


def fit_resnet18(hparams: Dict[str, float],
                 device: Union[torch.device, int],
                 tracker: Tracker,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 classes: int,
                 max_epochs: int) -> Dict[str, float]:
    # Model preparation
    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True, verbose=False)

    old_fc = model.fc
    model.fc = nn.Linear(in_features=old_fc.in_features, out_features=classes)

    model.type(torch.cuda.FloatTensor)
    model.to(device)

    loss = nn.CrossEntropyLoss()

    params_old = list([param for name, param in model.named_parameters() if not name.startswith('fc.')])
    params_new = list([param for name, param in model.named_parameters() if name.startswith('fc.')])
    optimizer_params = [
        {'params': params_old, 'lr': hparams['lrA'], 'weight_decay': hparams['wdA']},
        {'params': params_new, 'lr': hparams['lrB'], 'weight_decay': hparams['wdB']}
    ]
    optimizer = optim.AdamW(optimizer_params)

    # Training setup
    trial_tracker = tracker.new_trial(hparams)
    fit(device, model, train_loader, val_loader, loss, optimizer, num_epochs=max_epochs, tracker=trial_tracker, display_progress=False)

    # Reporting
    metrics_df = trial_tracker.metrics_df
    results = {
        'hp/best_val_acc': metrics_df['val_acc'].max(),
        'hp/best_val_loss': metrics_df['val_loss'].min(),
        'hp/final_val_acc': metrics_df['val_acc'].iloc[-1],
        'hp/final_val_loss': metrics_df['val_loss'].iloc[-1],

        'hp/best_train_acc': metrics_df['train_acc'].max(),
        'hp/best_train_loss': metrics_df['train_loss'].min(),
        'hp/final_train_acc': metrics_df['train_acc'].iloc[-1],
        'hp/final_train_loss': metrics_df['train_loss'].iloc[-1],
    }
    trial_tracker.trial_completed(results)

    return results


def snapshot_config(config: Dict) -> SnapshotConfig:
    return SnapshotConfig(root_dir=config['tracker']['snapshot_root'],
                          snapshotters=[
                              CheckpointSnapshotter(frequency=20),
                              BestCVSnapshotter(metric_name='acc', asc=False, top=5),
                              BestCVSnapshotter(metric_name='loss', asc=True, top=5)])


def tensorboard_config(config: Dict) -> TensorboardConfig:
    return TensorboardConfig(root_dir=config['tracker']['tensorboard_root'])


@click.command(name='search')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('-t', '--trials', 'trials', required=True, help='number of trials to perform', type=int)
@click.option('-m', '--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
def search(experiment: str, device_name: str, trials: int, max_epochs: int):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)
    tensorboard_cfg = tensorboard_config(config)

    tracker = Tracker(snapshot_cfg, tensorboard_cfg, experiment=experiment)
    launch_tensorboard(tracker.tensorboard_dir)

    pipeline_data = load_data(config['datasets'], config['db'])

    stack_bundle, model_bundle = split_eval_main_data(pipeline_data.train, 0.1)
    val_bundle, train_bundle = split_eval_main_data(model_bundle, 0.2)

    def fit_trial_resnet18(hparams: Dict[str, float]):
        results = fit_resnet18(hparams,
                               device=device,
                               tracker=tracker,
                               train_loader=train_bundle.loader,
                               val_loader=val_bundle.loader,
                               classes=train_bundle.dataset.wrapped.classes(),
                               max_epochs=max_epochs)

        return results['hp/best_val_loss']

    space = {
        'resnet18': {
            'lrA': hp.loguniform('lrA', math.log(1e-5), math.log(1)),
            'wdA': hp.loguniform('wdA', math.log(1e-4), math.log(1)),
            'lrB': hp.loguniform('lrB', math.log(1e-5), math.log(1)),
            'wdB': hp.loguniform('wdB', math.log(1e-4), math.log(1))
        }
    }

    fmin(fit_trial_resnet18, space=space['resnet18'], algo=hyperopt.rand.suggest, max_evals=trials)

    input("\nPipeline completed, press Enter to exit (this will terminate TensorBoard)\n")


@click.command(name='evaluate')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('--top', 'top', default=10, help='number of best trials to show', type=int)
def evaluate(experiment: str, device_name: str, top: int):
    def dump(df: pd.DataFrame) -> str:
        df = df.drop(columns=['experiment', 'trial', 'time', 'directory'])
        df = df.head(top)
        return tabulate(df, headers="keys", showindex=False, tablefmt='github')

    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        metric_name = 'val_acc'
        metric_asc = False
        return df.sort_values(by=metric_name, ascending=metric_asc)

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)

    df_res = Tracker.load_stats(snapshot_cfg, experiment)
    df_snap = df_res[df_res['snapshot'] != '']

    df_final = metric_sort(df_snap[df_snap['epoch'] == df_snap['num_epochs']])
    df_best = metric_sort(df_snap.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))

    print(f'Final results by trial: \n\n{dump(df_final)}\n\n\n')
    print(f'Best results by trial: \n\n{dump(df_best)}\n\n\n')

    row = df_final.head(1).to_dict(orient='records')[0]
    snapshot = Snapshotter.load_snapshot(row['directory'], row['snapshot'])

    pipeline_data = load_data(config['datasets'], config['db'])
    model = snapshot.model
    loss = snapshot.loss
    model.to(device)
    print(score(device, model, pipeline_data.public_test.loader, loss))


@click.group()
def cli():
    pass


if __name__ == "__main__":
    PIPELINE_LOGGER = 'pipeline.airliners'

    cli.add_command(search)
    cli.add_command(evaluate)
    cli()
