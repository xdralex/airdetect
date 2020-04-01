import os
import re
from typing import Optional

import click
import pandas as pd
import torch
import yaml
from torch import nn
from torchsummary import summary

from pipelines.aircraft_classification import cli_search as cls_search
from pipelines.aircraft_classification import cli_trial as cls_trial
from pipelines.aircraft_classification import cli_eval as cls_eval
from pipelines.util import dump, launch_tensorboard
from wheel5.introspection import introspect, make_dot
from wheel5.tracking import Tracker, TrialTracker
from wheel5 import logutils


@click.command(name='introspect-nn')
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('-s', '--shape', 'shape', required=True, help='input shape N x a_1 x a_2 x ... x a_k (e.g. 4x3x224x224)', type=str)
@click.option('-d', '--device', 'device', default='cuda:0', help='device number (0, 1, ...)', type=int)
def cli_introspect_nn(repo: str, network: str, shape: str, device: int):
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
    device = torch.device(f'cuda:{device}')
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


@click.command(name='list-experiments')
def cli_list_experiments():
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    tracker_root = config['tracking']['tracker_root']
    df_experiments = Tracker.load_experiment_stats(tracker_root)

    print(f'Experiments: \n\n{dump(df_experiments.sort_values(by="experiment"))}\n\n\n')


@click.command(name='list-trials')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('--top', 'top', default=10, type=int, help='number of top trials to show (default: 10')
@click.option('--metric', 'metric_name', required=True, type=str, help='name of the metric to sort by')
@click.option('--order', 'order', required=True, type=click.Choice(['asc', 'desc']), help='ascending/descending sort order')
@click.option('--hide', 'hide', default='experiment,trial', type=str, help='columns to hide')
@click.option('--incomplete', 'incomplete', is_flag=True, help='load incomplete trials')
def cli_list_trials(experiment: str, top: int, metric_name: str, order: str, hide: str, incomplete: bool):
    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=metric_name, ascending=(order == 'asc'))

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    tracker_root = config['tracking']['tracker_root']
    df_res = Tracker.load_trial_stats(tracker_root, experiment, complete_only=not incomplete)

    if df_res is None:
        print('No completed trials found')
    else:
        drop_cols = [col.strip() for col in hide.split(',') if col.strip() != '']
        df_best = metric_sort(df_res.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))
        print(f'Best results by trial: \n\n{dump(df_best, top=top, drop_cols=drop_cols)}\n\n\n')


@click.command(name='dump-trial')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('-t', '--trial', 'trial', required=True, type=str, help='trial name')
@click.option('--hide', 'hide', default='experiment,trial', type=str, help='columns to hide')
@click.option('--incomplete', 'incomplete', is_flag=True, help='load incomplete trials')
def cli_dump_trial(experiment: str, trial: str, hide: str, incomplete: bool):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    tracker_root = config['tracking']['tracker_root']
    df_res = TrialTracker.load_trial_stats(tracker_root, experiment, trial, load_hparams=False, complete_only=not incomplete)

    if df_res is None:
        print('Trial does not have any saved metrics')
    else:
        drop_cols = [col.strip() for col in hide.split(',') if col.strip() != '']
        print(f'Results: \n\n{dump(df_res, drop_cols=drop_cols)}\n\n\n')


@click.command(name='tensorboard')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
def cli_tensorboard(experiment: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    tensorboard_root = config['tracking']['tensorboard_root']
    tensorboard_dir = os.path.join(tensorboard_root, experiment)

    launch_tensorboard(tensorboard_dir)

    input("\nPress Enter to exit (this will terminate TensorBoard)\n")


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(click.command(name='cls-trial')(cls_trial))
    cli.add_command(click.command(name='cls-search')(cls_search))
    cli.add_command(click.command(name='cls-eval')(cls_eval))

    cli.add_command(cli_introspect_nn)

    cli.add_command(cli_list_experiments)
    cli.add_command(cli_list_trials)
    cli.add_command(cli_dump_trial)

    cli.add_command(cli_tensorboard)

    cli()
