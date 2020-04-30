import os

import click
import hyperopt
import pandas as pd
import yaml
from hyperopt import fmin
from typing import Dict
from wheel5 import logutils
from wheel5.tracking import Tracker, CheckpointPattern

from .pipeline import AircraftClassifierConfig
from .tools import fit_trial, eval_blend, build_heatmaps
from .search import make_space_dict
from ..util import parse_kv, dump


@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device', default='0', help='device number (0, 1, ...)', type=int)
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('--train-wrk', 'train_workers', default=4, help='number of train dataloader workers', type=int)
@click.option('--eval-wrk', 'eval_workers', default=4, help='number of eval dataloader workers', type=int)
@click.option('--rnd-seed', 'rnd_seed', default=42, help='random seed', type=int)
@click.option('--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
@click.option('--kv', 'kv', default='', help='key-value parameters (k1=v1,k2=v2,...)', type=str)
def cli_trial(experiment: str, device: int, repo: str, network: str,
              train_workers: int, eval_workers: int,
              rnd_seed: int, max_epochs: int, kv: str):

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    snapshot_root = config['tracking']['snapshot_root']
    tensorboard_root = config['tracking']['tensorboard_root']
    tracker_root = config['tracking']['tracker_root']

    snapshot_dir = os.path.join(snapshot_root, experiment)
    tracker = Tracker(tracker_root, experiment)

    pipeline_config = AircraftClassifierConfig(
        random_state_seed=rnd_seed,

        classes_path=config['datasets']['classes'],
        dataset_config=config['datasets']['train'],
        heatmaps_path=config['boost']['heatmaps']['train'],

        repo=repo,
        network=network,

        kv=parse_kv(kv),

        train_workers=train_workers,
        eval_workers=eval_workers
    )

    print(f'\n{Tracker.dict_to_key(pipeline_config.kv)}')
    results = fit_trial(tracker=tracker,
                        snapshot_dir=snapshot_dir,
                        tensorboard_root=tensorboard_root,
                        experiment=experiment,
                        device=device,
                        config=pipeline_config,
                        max_epochs=max_epochs)

    print()
    for k, v in results.items():
        print(f'{k} = {v:.5f}')


@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device', default='0', help='device number (0, 1, ...)', type=int)
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('--train-wrk', 'train_workers', default=4, help='number of train dataloader workers', type=int)
@click.option('--eval-wrk', 'eval_workers', default=4, help='number of eval dataloader workers', type=int)
@click.option('--rnd-seed', 'rnd_seed', default=42, help='random seed', type=int)
@click.option('--space', 'space', required=True, help='search space name', type=str)
@click.option('--trials', 'trials', required=True, help='number of trials to perform', type=int)
@click.option('--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
def cli_search(experiment: str, device: int, repo: str, network: str,
               train_workers: int, eval_workers: int,
               rnd_seed: int, space: str, trials: int, max_epochs: int):

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    snapshot_root = config['tracking']['snapshot_root']
    tensorboard_root = config['tracking']['tensorboard_root']
    tracker_root = config['tracking']['tracker_root']

    snapshot_dir = os.path.join(snapshot_root, experiment)
    tracker = Tracker(tracker_root, experiment)

    def fit_trial_wrapper(kv: Dict[str, float]):
        pipeline_config = AircraftClassifierConfig(
            random_state_seed=rnd_seed,

            classes_path=config['datasets']['classes'],
            dataset_config=config['datasets']['train'],
            heatmaps_path=config['boost']['heatmaps']['train'],

            repo=repo,
            network=network,

            kv=kv,

            train_workers=train_workers,
            eval_workers=eval_workers
        )

        print(f'\n{Tracker.dict_to_key(pipeline_config.kv)}')
        results = fit_trial(tracker=tracker,
                            snapshot_dir=snapshot_dir,
                            tensorboard_root=tensorboard_root,
                            experiment=experiment,
                            device=device,
                            config=pipeline_config,
                            max_epochs=max_epochs)

        return results['min_val_loss']

    space_dict = make_space_dict()
    fmin(fit_trial_wrapper, space=space_dict[space], algo=hyperopt.rand.suggest, max_evals=trials, verbose=False, show_progressbar=False)


@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device', default='0', help='device number (0, 1, ...)', type=int)
@click.option('--top', 'top', default=1, help='number of top models to use', type=int)
@click.option('--metric', 'metric_name', required=True, type=str, help='name of the metric to sort by')
@click.option('--order', 'order', required=True, type=click.Choice(['asc', 'desc']), help='ascending/descending sort order')
@click.option('--test', 'test', default='public', type=click.Choice(['public', 'private']), help='public/private test dataset')
@click.option('--hide', 'hide', default='experiment,trial,lr_f,lr_t0,lr_w', type=str, help='columns to hide')
@click.option('--incomplete', 'incomplete', is_flag=True, help='load incomplete trials')
def cli_eval(experiment: str, device: int, top: int, metric_name: str, order: str, test: str, hide: str, incomplete: bool):
    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=metric_name, ascending=(order == 'asc'))

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_root = config['tracking']['snapshot_root']
    tracker_root = config['tracking']['tracker_root']

    df_res = Tracker.load_trial_stats(tracker_root, experiment, complete_only=not incomplete)

    if df_res is None:
        print('No completed trials found')
    else:
        drop_cols = [col.strip() for col in hide.split(',') if col.strip() != '']
        df_best = metric_sort(df_res.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))
        df_best = df_best.head(top)
        df_best.insert(0, 'n', range(0, len(df_best)))

        df_best_dump = dump(df_best, drop_cols=drop_cols)
        print(f'Evaluating blend of top {top} models on the >>{test}<< test dataset: \n\n{df_best_dump}\n\n\n')

        snapshot_dir = os.path.join(snapshot_root, experiment)

        def snapshot_path(entry):
            return CheckpointPattern.path(os.path.join(snapshot_dir, entry.trial), entry.epoch)

        results = eval_blend(dataset_config=config['datasets'][f'{test}_test'],
                             device=device,
                             snapshot_paths=[snapshot_path(entry) for entry in df_best.itertuples()])

        print()
        for k, v in results.items():
            print(f'{k} = {v:.5f}')


@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device', default='0', help='device number (0, 1, ...)', type=int)
@click.option('--top', 'top', default=1, help='number of top models to use', type=int)
@click.option('--metric', 'metric_name', required=True, type=str, help='name of the metric to sort by')
@click.option('--order', 'order', required=True, type=click.Choice(['asc', 'desc']), help='ascending/descending sort order')
@click.option('--index', 'index', default=0, type=int, help='model index (0, 1, ...)')
@click.option('--hide', 'hide', default='experiment,trial,lr_f,lr_t0,lr_w', type=str, help='columns to hide')
@click.option('--incomplete', 'incomplete', is_flag=True, help='load incomplete trials')
def cli_build_heatmaps(experiment: str, device: int, top: int, metric_name: str, order: str, index: int, hide: str, incomplete: bool):
    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=metric_name, ascending=(order == 'asc'))

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_root = config['tracking']['snapshot_root']
    tracker_root = config['tracking']['tracker_root']

    df_res = Tracker.load_trial_stats(tracker_root, experiment, complete_only=not incomplete)

    if df_res is None:
        print('No completed trials found')
    else:
        drop_cols = [col.strip() for col in hide.split(',') if col.strip() != '']
        df_best = metric_sort(df_res.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))
        df_best = df_best.head(top)
        df_best.insert(0, 'n', range(0, len(df_best)))

        df_best_dump = dump(df_best, drop_cols=drop_cols)
        print(f'Top {top} models: \n\n{df_best_dump}\n\n\n')

        snapshot_dir = os.path.join(snapshot_root, experiment)
        entry = list(df_best.itertuples())[index]
        snapshot_path = CheckpointPattern.path(os.path.join(snapshot_dir, entry.trial), entry.epoch)

        print(f'Using trial [{index}]: {entry.trial}')
        build_heatmaps(dataset_config=config['datasets'][f'train'],
                       snapshot_path=snapshot_path,
                       heatmap_path=config['boost']['heatmaps']['train'],
                       device=device)
