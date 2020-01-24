import logging
import math
from typing import Dict, NamedTuple, List

import hyperopt
import numpy as np
import torch
import yaml
from hyperopt import hp, fmin
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tensorboard import program

from data import load_aircraft_data
from wheel5 import logutils
from wheel5.dataset import LMDBImageDataset, WrappingTransformDataset, split_indices
from wheel5.model import fit
from wheel5.organizer import Organizer
from wheel5.snapshotters import CheckpointSnapshotter, BestCVSnapshotter
from wheel5.transforms import SquarePaddedResize


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
    tb.configure(argv=[None, '--bind_all', '--port', f'{port}',  '--logdir', tensorboard_root])
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


def fit_resnet18(hparams: Dict[str, float], train_loader: DataLoader, val_loader: DataLoader, classes: int) -> Dict[str, float]:
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
    org_trial = org.new_trial(hparams=hparams)
    snapshot_dir = org_trial.snapshot_dir()
    tensorboard_dir = org_trial.tensorboard_dir()

    tb_writer = SummaryWriter(tensorboard_dir, max_queue=100, flush_secs=60)

    metrics_df = fit(device, model, train_loader, val_loader, loss, optimizer,
                     num_epochs=20,
                     snapshotter=[
                         CheckpointSnapshotter(snapshot_dir, frequency=10),
                         BestCVSnapshotter(snapshot_dir, metric_name='accuracy', asc=False, best=3),
                         BestCVSnapshotter(snapshot_dir, metric_name='loss', asc=True, best=3),
                     ],
                     tb_writer=tb_writer,
                     display_progress=False)

    # Reporting
    results = {
        'hp/best_val_acc': metrics_df['val_accuracy'].max(),
        'hp/best_val_loss': metrics_df['val_loss'].min(),
        'hp/final_val_acc': metrics_df['val_accuracy'].iloc[-1],
        'hp/final_val_loss': metrics_df['val_loss'].iloc[-1],

        'hp/best_train_acc': metrics_df['train_accuracy'].max(),
        'hp/best_train_loss': metrics_df['train_loss'].min(),
        'hp/final_train_acc': metrics_df['train_accuracy'].iloc[-1],
        'hp/final_train_loss': metrics_df['train_loss'].iloc[-1],
    }

    tb_writer.add_hparams(hparams, results)
    tb_writer.flush()

    return results


# Pipeline
if __name__ == "__main__":
    PIPELINE_LOGGER = 'pipeline.airliners'

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    logutils.configure_logging(config['logging'])
    device = torch.device('cuda:0')

    org = Organizer(experiment='resnet18_hp1', **config['organizer'])

    launch_tensorboard(org.tensorboard_experiment())

    pipeline_data = load_data(config['datasets'], config['db'])

    stack_bundle, model_bundle = split_eval_main_data(pipeline_data.train, 0.1)
    val_bundle, train_bundle = split_eval_main_data(model_bundle, 0.2)


    def fit_trial_resnet18(hparams: Dict[str, float]):
        results = fit_resnet18(hparams,
                               train_loader=train_bundle.loader,
                               val_loader=val_bundle.loader,
                               classes=train_bundle.dataset.wrapped.classes())

        return results['hp/best_val_loss']


    space = {
        'resnet18': {
            'lrA': hp.loguniform('lrA', math.log(1e-5), math.log(1)),
            'wdA': hp.loguniform('wdA', math.log(1e-4), math.log(1)),
            'lrB': hp.loguniform('lrB', math.log(1e-5), math.log(1)),
            'wdB': hp.loguniform('wdB', math.log(1e-4), math.log(1))
        }
    }

    best = fmin(fit_trial_resnet18, space=space['resnet18'], algo=hyperopt.rand.suggest, max_evals=30)
