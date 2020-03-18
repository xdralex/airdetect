import json
import math
import os
import re
from typing import Dict, Optional

import click
import hyperopt
import pandas as pd
import torch
import yaml
from hyperopt import hp, fmin
from torch import nn
from torchsummary import summary
from wheel5 import logutils
from wheel5.introspection import introspect, make_dot
from wheel5.tracking import Tracker, TrialTracker

from experiments import ExperimentConfig, fit_model, score_model_blend, TRANSFORMS_BUNDLE
from util import launch_tensorboard, dump, snapshot_config, tensorboard_config


@click.command(name='search')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('--space', 'space', required=True, help='search space name', type=str)
@click.option('--trials', 'trials', required=True, help='number of trials to perform', type=int)
@click.option('--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
@click.option('--freeze', 'freeze', default=-1, help='freeze first K layers (set to negative or zero to disable)', type=int)
@click.option('--mixup', 'mixup', is_flag=True, help='apply mixup augmentation', type=bool)
@click.option('--cutmix', 'cutmix', is_flag=True, help='apply cutmix augmentation', type=bool)
def cli_search(experiment: str, device_name: str, repo: str, network: str, space: str, trials: int, max_epochs: int, freeze: int, mixup: bool, cutmix: bool):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)
    tensorboard_cfg = tensorboard_config(config)

    tracker = Tracker(snapshot_cfg, tensorboard_cfg, experiment=experiment, sample_img_transform=TRANSFORMS_BUNDLE.sample)
    launch_tensorboard(tracker.tensorboard_dir)

    def fit_model_trial(hparams: Dict[str, float]):
        experiment_config = ExperimentConfig(repo=repo,
                                             network=network,
                                             hparams=hparams,
                                             max_epochs=max_epochs,
                                             freeze=freeze,
                                             mixup=mixup,
                                             cutmix=cutmix)

        results = fit_model(dataset_config=config['datasets']['train'],
                            experiment_config=experiment_config,
                            device=device,
                            tracker=tracker,
                            debug=False,
                            display_progress=False)

        return results['hp/best_val_loss']

    space_dict = {
        'resnet50_narrow': {
            'lrA': hp.uniform('lrA', 2e-4, 4e-4),
            'wdA': hp.loguniform('wdA', math.log(1e-2), math.log(1)),
            'lrB': hp.uniform('lrB', 2e-4, 4e-4),
            'wdB': hp.loguniform('wdB', math.log(1e-2), math.log(1)),
            'cos_t0': hp.uniform('cos_t0', 9.999, 10.001),
            'cos_f': hp.uniform('cos_f', 1.999, 2.001),
            'smooth': hp.loguniform('smooth', math.log(1e-4), math.log(1)),
            'alpha': hp.uniform('alpha', 0.1, 1.0)
        },
        'resnet50_wide': {
            'lrA': hp.loguniform('lrA', math.log(1e-4), math.log(1e-2)),
            'wdA': hp.loguniform('wdA', math.log(1e-3), math.log(1e+1)),
            'lrB': hp.loguniform('lrB', math.log(1e-4), math.log(1e-2)),
            'wdB': hp.loguniform('wdB', math.log(1e-3), math.log(1e+1)),
            'cos_t0': hp.uniform('cos_t0', 9.999, 10.001),
            'cos_f': hp.uniform('cos_f', 1.999, 2.001),
            'smooth': hp.loguniform('smooth', math.log(1e-4), math.log(1)),
            'alpha': hp.loguniform('alpha', math.log(1e-3), math.log(1e+1))
        },

        'resnet101_narrow': {
            'lrA': hp.uniform('lrA', 1e-4, 4e-4),
            'wdA': hp.uniform('wdA', 1e-1, 1.0),
            'lrB': hp.uniform('lrB', 1e-4, 4e-4),
            'wdB': hp.uniform('wdB', 1e-1, 1.0),
            'cos_t0': hp.uniform('cos_t0', 9.999, 10.001),
            'cos_f': hp.uniform('cos_f', 1.999, 2.001),
            'smooth': hp.loguniform('smooth', math.log(1e-4), math.log(1))
        },
        'resnet101_wide': {
            'lrA': hp.loguniform('lrA', math.log(1e-4), math.log(1e-3)),
            'wdA': hp.loguniform('wdA', math.log(1e-2), math.log(1e+1)),
            'lrB': hp.loguniform('lrB', math.log(1e-4), math.log(1e-3)),
            'wdB': hp.loguniform('wdB', math.log(1e-2), math.log(1e+1)),
            'cos_t0': hp.uniform('cos_t0', 9.999, 10.001),
            'cos_f': hp.uniform('cos_f', 1.999, 2.001),
            'smooth': hp.loguniform('smooth', math.log(1e-4), math.log(1))
        },

        'se_resnet50_narrow': {
            'lrA': hp.uniform('lrA', 2e-4, 4e-4),
            'wdA': hp.uniform('wdA', 1e-1, 1.0),
            'lrB': hp.uniform('lrB', 2e-4, 4e-4),
            'wdB': hp.uniform('wdB', 1e-1, 1.0),
            'cos_t0': hp.uniform('cos_t0', 9.999, 10.001),
            'cos_f': hp.uniform('cos_f', 1.999, 2.001),
            'smooth': hp.loguniform('smooth', math.log(1e-4), math.log(1))
        },
        'se_resnet50_wide': {
            'lrA': hp.loguniform('lrA', math.log(1e-4), math.log(1e-2)),
            'wdA': hp.loguniform('wdA', math.log(1e-2), math.log(1e+1)),
            'lrB': hp.loguniform('lrB', math.log(1e-4), math.log(1e-2)),
            'wdB': hp.loguniform('wdB', math.log(1e-2), math.log(1e+1)),
            'cos_t0': hp.uniform('cos_t0', 10, 20),
            'cos_f': hp.uniform('cos_f', 1, 2),
            'smooth': hp.loguniform('smooth', math.log(1e-4), math.log(1))
        }
    }

    fmin(fit_model_trial, space=space_dict[space], algo=hyperopt.rand.suggest, max_evals=trials)

    input("\nSearch completed, press Enter to exit (this will terminate TensorBoard)\n")


@click.command(name='trial')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
@click.option('--freeze', 'freeze', default=-1, help='freeze first K layers', type=int)
@click.option('--mixup', 'mixup', is_flag=True, help='apply mixup augmentation', type=bool)
@click.option('--cutmix', 'cutmix', is_flag=True, help='apply cutmix augmentation', type=bool)
def cli_trial(experiment: str, device_name: str, repo: str, network: str, max_epochs: int, freeze: int, mixup: bool, cutmix: bool):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)
    tensorboard_cfg = tensorboard_config(config)

    tracker = Tracker(snapshot_cfg, tensorboard_cfg, experiment=experiment, sample_img_transform=TRANSFORMS_BUNDLE.sample)
    launch_tensorboard(tracker.tensorboard_dir)

    hparams = {
        'lrA': 0.0003,
        'lrB': 0.0003,
        'wdA': 0.1,
        'wdB': 0.1,
        'cos_t0': 10,
        'cos_f': 2,
        'smooth': 0.0,
        'alpha': 0.1
    }

    experiment_config = ExperimentConfig(repo=repo,
                                         network=network,
                                         hparams=hparams,
                                         max_epochs=max_epochs,
                                         freeze=freeze,
                                         mixup=mixup,
                                         cutmix=cutmix)

    results = fit_model(dataset_config=config['datasets']['train'],
                        experiment_config=experiment_config,
                        device=device,
                        tracker=tracker,
                        debug=True,
                        display_progress=True)

    print(json.dumps(results, indent=4))

    input("\nTrial completed, press Enter to exit (this will terminate TensorBoard)\n")


@click.command(name='eval-top-blend')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('--kind', 'kind', required=True, type=click.Choice(['final', 'best']), help='use final/best models')
@click.option('--top', 'top', default=1, help='number of best models to use', type=int)
@click.option('--metric', 'metric_name', required=True, type=str, help='name of the metric to sort by')
@click.option('--order', 'order', required=True, type=click.Choice(['asc', 'desc']), help='ascending/descending sort order')
@click.option('--test', 'test', default='public', type=click.Choice(['public', 'private']), help='public/private test dataset')
@click.option('--hide', 'hide', default='experiment,trial,time,directory,ctrl_loss,ctrl_acc,snapshot', type=str, help='columns to hide')
def cli_eval_top_blend(experiment: str, device_name: str, kind: str, top: int, metric_name: str, order: str, test: str, hide: str):
    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=metric_name, ascending=(order == 'asc'))

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)
    df_res = Tracker.load_trial_stats(snapshot_cfg, experiment)
    df_res = df_res[df_res['snapshot'] != '']

    if kind == 'final':
        df_model = metric_sort(df_res[df_res['epoch'] == df_res['num_epochs']])
    elif kind == 'best':
        df_model = metric_sort(df_res.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))
    else:
        raise click.BadOptionUsage('kind', f'Unsupported kind option: "{kind}"')

    df_top_models = df_model.head(top)
    drop_cols = [col.strip() for col in hide.split(',')]
    print(f'Averaging top models: \n\n{dump(df_top_models, drop_cols=drop_cols)}\n\n\n')

    print(f'Evaluating model performance on the >>{test}<< test dataset:\n')
    paths = [(row.directory, row.snapshot) for row in df_top_models.head(top).itertuples()]
    results = score_model_blend(dataset_config=config['datasets'][f'{test}_test'],
                                device=device,
                                paths=paths)
    print(results)


@click.command(name='introspect-nn')
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('-s', '--shape', 'shape', required=True, help='input shape N x a_1 x a_2 x ... x a_k (e.g. 4x3x224x224)', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
def cli_introspect_nn(repo: str, network: str, shape: str, device_name: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        viz_nn_dir = config['viz']['nn_dir']

    dims = []
    for dim_str in shape.split('x'):
        try:
            dims.append(int(dim_str.strip()))
        except ValueError:
            raise click.BadOptionUsage('shape', f'Invalid input shape: "{shape}" - dimension "{dim_str}"')
    if len(dims) <= 1:
        raise click.BadOptionUsage('shape', f'Invalid input shape: "{shape}": at least two dimensions (batch size and tensor size) must be provided')

    model = torch.hub.load(repo, network, pretrained=True, verbose=False)
    device = torch.device(device_name)
    model = model.to(device)

    def print_model_params(module_name: Optional[str], module: nn.Module, depth: int = 0):
        indent: str = ' ' * 4

        module_name_qual = f'{module_name}: ' if module_name else ''
        module_type = type(module)
        print(f'{indent * depth}{module_name_qual}{module_type.__module__}.{module_type.__name__}')

        depth = depth + 1

        for name, param in module.named_parameters(recurse=False):
            param_name_qual = f'{module_name}.{name}' if module_name else f'{name}'
            param_shape = 'x'.join([str(dim) for dim in param.shape])
            print(f'{indent * depth}{param_name_qual} - {param_shape}')

        for name, child in module.named_children():
            child_name_qual = f'{module_name}.{name}' if module_name else f'{name}'
            print_model_params(f'{indent * depth}{child_name_qual}', child, depth)

    print(f'Model:\n{model}\n\n')

    print(f'Parameters:\n')
    print_model_params(module_name=None, module=model)
    print('\n')

    print(f'Summary:\n')
    summary(model, input_size=tuple(dims[1:]), batch_size=dims[0])
    print('\n')

    graph = introspect(model, input_size=dims)
    dot = make_dot(graph)

    filename = f'{repo}-{network}'
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    dot.render(filename=filename, directory=viz_nn_dir, format='dot')

    print(f'Dot file saved to: {os.path.join(viz_nn_dir, filename)}.dot')


@click.command(name='list-top-trials')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('--top', 'top', default=10, type=int, help='number of top trials to show (default: 10')
@click.option('--metric', 'metric_name', required=True, type=str, help='name of the metric to sort by')
@click.option('--order', 'order', required=True, type=click.Choice(['asc', 'desc']), help='ascending/descending sort order')
@click.option('--all/--snap', 'list_all', default=False, help='list all entries (default) / only entries with snapshots')
@click.option('--hide', 'hide', default='experiment,trial,time,directory,ctrl_loss,ctrl_acc,snapshot', type=str, help='columns to hide')
def cli_list_top_trials(experiment: str, top: int, metric_name: str, order: str, list_all: bool, hide: str):
    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=metric_name, ascending=(order == 'asc'))

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_cfg = snapshot_config(config)
    df_res = Tracker.load_trial_stats(snapshot_cfg, experiment)

    if df_res is None:
        print('No completed trials found')
    else:
        drop_cols = [col.strip() for col in hide.split(',')]

        df_snap = df_res if list_all else df_res[df_res['snapshot'] != '']

        df_final = metric_sort(df_snap[df_snap['epoch'] == df_snap['num_epochs']])
        print(f'Final results by trial: \n\n{dump(df_final, top=top, drop_cols=drop_cols)}\n\n\n')

        df_best = metric_sort(df_snap.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))
        print(f'Best results by trial: \n\n{dump(df_best, top=top, drop_cols=drop_cols)}\n\n\n')


@click.command(name='list-all-trials')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('--all/--snap', 'list_all', default=False, help='list all entries (default) / only entries with snapshots')
@click.option('--hide', 'hide', default='experiment,trial,time,directory,ctrl_loss,ctrl_acc,snapshot', type=str, help='columns to hide')
def cli_list_all_trials(experiment: str, list_all: bool, hide: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_cfg = snapshot_config(config)
    df_res = Tracker.load_trial_stats(snapshot_cfg, experiment)

    if df_res is None:
        print('No completed trials found')
    else:
        drop_cols = [col.strip() for col in hide.split(',')]

        df_snap = df_res if list_all else df_res[df_res['snapshot'] != '']
        print(f'Results: \n\n{dump(df_snap, drop_cols=drop_cols)}\n\n\n')


@click.command(name='list-trial')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('-t', '--trial', 'trial', required=True, type=str, help='trial name')
@click.option('--all/--snap', 'list_all', default=True, help='list all entries (default) / only entries with snapshots')
@click.option('--hide', 'hide', default='experiment,trial,time,directory,ctrl_loss,ctrl_acc,snapshot', type=str, help='columns to hide')
def cli_list_trial(experiment: str, trial: str, list_all: bool, hide: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_cfg = snapshot_config(config)
    df_res = TrialTracker.load_trial_stats(snapshot_cfg, experiment, trial, load_hparams=False, complete_only=False)

    if df_res is None:
        print('Trial does not have any saved metrics')
    else:
        drop_cols = [col.strip() for col in hide.split(',')]

        df_snap = df_res if list_all else df_res[df_res['snapshot'] != '']
        print(f'Results: \n\n{dump(df_snap, drop_cols=drop_cols)}\n\n\n')


@click.command(name='list-experiments')
def cli_list_experiments():
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_cfg = snapshot_config(config)
    df_experiments = Tracker.load_experiment_stats(snapshot_cfg)

    print(f'Experiments: \n\n{dump(df_experiments.sort_values(by="experiment"))}\n\n\n')


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(cli_trial)
    cli.add_command(cli_search)
    cli.add_command(cli_eval_top_blend)

    cli.add_command(cli_introspect_nn)

    cli.add_command(cli_list_top_trials)
    cli.add_command(cli_list_all_trials)
    cli.add_command(cli_list_trial)

    cli.add_command(cli_list_experiments)

    cli()
