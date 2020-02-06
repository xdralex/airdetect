import json
import math
import re
from typing import Dict
import os

import click
import hyperopt
import pandas as pd
import torch
import yaml
from hyperopt import hp, fmin
from torchsummary import summary
from wheel5 import logutils
from wheel5.dataset import split_eval_main_data
from wheel5.introspection import introspect, make_dot
from wheel5.model import score_blend
from wheel5.tracking import Tracker, Snapshotter

from experiments import prepare_data, fit_resnet
from util import launch_tensorboard, dump, snapshot_config, tensorboard_config


@click.command(name='introspect-nn')
@click.option('-r', '--repo', 'repo', default='pytorch/vision', help='repository (e.g. pytorch/vision)', type=str)
@click.option('-t', '--tag', 'tag', default='v0.4.2', help='tag (e.g. v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('-s', '--shape', 'shape', required=True, help='input shape N x a_1 x a_2 x ... x a_k (e.g. 4x3x224x224)', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
def cli_introspect_nn(repo: str, tag: str, network: str, shape: str, device_name: str):
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

    model = torch.hub.load(f'{repo}:{tag}', network, pretrained=True, verbose=False)
    device = torch.device(device_name)
    model = model.to(device)

    print(f'Model:\n{model}\n')

    print(f'Summary:\n')
    summary(model, input_size=tuple(dims[1:]), batch_size=dims[0])
    print('')

    graph = introspect(model, input_size=dims)
    dot = make_dot(graph)

    filename = f'{repo}-{tag}-{network}'
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    dot.render(filename=filename, directory=viz_nn_dir, format='dot')

    print(f'Dot file saved to: {os.path.join(viz_nn_dir, filename)}.dot')


@click.command(name='trial')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('-m', '--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
def cli_trial(experiment: str, device_name: str, max_epochs: int):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)
    tensorboard_cfg = tensorboard_config(config)

    tracker = Tracker(snapshot_cfg, tensorboard_cfg, experiment=experiment)
    launch_tensorboard(tracker.tensorboard_dir)

    pipeline_data = prepare_data(config['datasets'])

    val_bundle, train_bundle = split_eval_main_data(pipeline_data.train, 0.2)

    hparams = {
        'lrA': 0.000319403,
        'lrB': 2.38508e-05,
        'wdA': 0.000406034,
        'wdB': 0.0104068,
        'freeze': 4
    }

    results = fit_resnet('resnet50',
                         hparams,
                         device=device,
                         tracker=tracker,
                         train_loader=train_bundle.loader,
                         val_loader=val_bundle.loader,
                         classes=train_bundle.dataset.wrapped.wrapped.classes(),  # TODO
                         max_epochs=max_epochs,
                         display_progress=True)

    print(json.dumps(results, indent=4))

    input("\nTrial completed, press Enter to exit (this will terminate TensorBoard)\n")


@click.command(name='search')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('-t', '--trials', 'trials', required=True, help='number of trials to perform', type=int)
@click.option('-m', '--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
def cli_search(experiment: str, device_name: str, trials: int, max_epochs: int):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)
    tensorboard_cfg = tensorboard_config(config)

    tracker = Tracker(snapshot_cfg, tensorboard_cfg, experiment=experiment)
    launch_tensorboard(tracker.tensorboard_dir)

    pipeline_data = prepare_data(config['datasets'])

    val_bundle, train_bundle = split_eval_main_data(pipeline_data.train, 0.2)

    def fit_trial_resnet(hparams: Dict[str, float]):
        results = fit_resnet('resnet50',
                             hparams,
                             device=device,
                             tracker=tracker,
                             train_loader=train_bundle.loader,
                             val_loader=val_bundle.loader,
                             classes=train_bundle.dataset.wrapped.wrapped.classes(),
                             max_epochs=max_epochs,
                             display_progress=False)

        return results['hp/best_val_loss']

    space = {
        'resnet18_broad': {
            'lrA': hp.loguniform('lrA', math.log(1e-5), math.log(1)),
            'wdA': hp.loguniform('wdA', math.log(1e-4), math.log(1)),
            'lrB': hp.loguniform('lrB', math.log(1e-5), math.log(1)),
            'wdB': hp.loguniform('wdB', math.log(1e-4), math.log(1))
        },

        'resnet50_narrow': {
            'lrA': hp.loguniform('lrA', math.log(1e-5), math.log(1e-3)),
            'wdA': hp.loguniform('wdA', math.log(1e-4), math.log(1)),
            'lrB': hp.loguniform('lrB', math.log(1e-5), math.log(1e-2)),
            'wdB': hp.loguniform('wdB', math.log(1e-4), math.log(1))
        }
    }

    fmin(fit_trial_resnet, space=space['resnet50_narrow'], algo=hyperopt.rand.suggest, max_evals=trials)

    input("\nSearch completed, press Enter to exit (this will terminate TensorBoard)\n")


@click.command(name='eval-top-blend')
@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device_name', default='cuda:0', help='device name (cuda:0, cuda:1, ...)', type=str)
@click.option('--kind', 'kind', required=True, type=click.Choice(['final', 'best']), help='use final/best models')
@click.option('--top', 'top', default=1, help='number of best models to use', type=int)
@click.option('--metric', 'metric_name', required=True, type=str, help='name of the metric to sort by')
@click.option('--order', 'order', required=True, type=click.Choice(['asc', 'desc']), help='ascending/descending sort order')
@click.option('--test', 'test', default='public', type=click.Choice(['public', 'private']), help='public/private test dataset')
def cli_eval_top_blend(experiment: str, device_name: str, kind: str, top: int, metric_name: str, order: str, test: str):
    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=metric_name, ascending=(order == 'asc'))

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    device = torch.device(device_name)

    snapshot_cfg = snapshot_config(config)
    df_res = Tracker.load_stats(snapshot_cfg, experiment)

    if kind == 'final':
        df_model = metric_sort(df_res[df_res['epoch'] == df_res['num_epochs']])
    elif kind == 'best':
        df_model = metric_sort(df_res.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))
    else:
        raise click.BadOptionUsage('kind', f'Unsupported kind option: "{kind}"')

    pipeline_data = prepare_data(config['datasets'])
    if test == 'public':
        test_bundle = pipeline_data.public_test
    elif test == 'private':
        test_bundle = pipeline_data.private_test
    else:
        raise click.BadOptionUsage('test', f'Unsupported test option: "{test}"')

    df_top_models = df_model.head(top)
    print(f'Averaging top models: \n\n{dump(df_top_models)}\n\n\n')

    models = []
    loss = None
    for row in df_top_models.head(top).itertuples():
        snapshot = Snapshotter.load_snapshot(row.directory, row.snapshot)
        model = snapshot.model.cpu()

        models.append(model)
        loss = snapshot.loss

        del snapshot

    print(f'Evaluating model performance on the >>{test}<< test dataset:\n')
    print(score_blend(device, models, test_bundle.loader, loss))


@click.command(name='list-top')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('--top', 'top', default=10, type=int, help='number of top trials to show (default: 10')
@click.option('--metric', 'metric_name', required=True, type=str, help='name of the metric to sort by')
@click.option('--order', 'order', required=True, type=click.Choice(['asc', 'desc']), help='ascending/descending sort order')
def cli_list_top(experiment: str, top: int, metric_name: str, order: str):
    def metric_sort(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=metric_name, ascending=(order == 'asc'))

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_cfg = snapshot_config(config)
    df_res = Tracker.load_stats(snapshot_cfg, experiment)

    if df_res is None:
        print('No completed trials found')
    else:
        df_final = metric_sort(df_res[df_res['epoch'] == df_res['num_epochs']])
        print(f'Final results by trial: \n\n{dump(df_final, top=top)}\n\n\n')

        df_best = metric_sort(df_res.groupby(['experiment', 'trial']).apply(lambda df: metric_sort(df).head(1)).reset_index(drop=True))
        print(f'Best results by trial: \n\n{dump(df_best, top=top)}\n\n\n')


@click.command(name='list-all')
@click.option('-e', '--experiment', 'experiment', required=True, type=str, help='experiment name')
@click.option('--all/--snap', 'list_all', default=True, help='list all entries (default) / only entries with snapshots')
def cli_list_all(experiment: str, list_all: bool):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    snapshot_cfg = snapshot_config(config)
    df_res = Tracker.load_stats(snapshot_cfg, experiment)

    if df_res is None:
        print('No completed trials found')
    else:
        if list_all:
            print(f'Results: \n\n{dump(df_res)}\n\n\n')
        else:
            df_snap = df_res[df_res['snapshot'] != '']
            print(f'Results with snapshots: \n\n{dump(df_snap)}\n\n\n')


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(cli_introspect_nn)
    cli.add_command(cli_trial)
    cli.add_command(cli_search)
    cli.add_command(cli_eval_top_blend)
    cli.add_command(cli_list_top)
    cli.add_command(cli_list_all)
    cli()
