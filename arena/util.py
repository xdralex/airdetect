import logging
from typing import Dict, List, Optional

import pandas as pd
from tabulate import tabulate
from tensorboard import program

from pipelines.aircraft_classification import DatasetConfig
from wheel5.tracking import SnapshotConfig, TensorboardConfig, CheckpointSnapshotter, BestCVSnapshotter


def launch_tensorboard(tensorboard_root: str, port: int = 6006):
    logger = logging.getLogger('arena.util')

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--bind_all', '--port', f'{port}', '--logdir', tensorboard_root])
    url = tb.launch()

    logger.info(f'Launched TensorBoard at {url}')


def snapshot_config(config: Dict) -> SnapshotConfig:
    config = config['tracker']['snapshot']

    snapshotters = [
        CheckpointSnapshotter(frequency=20),
        BestCVSnapshotter(metric_name='acc', asc=False, top=1),
        BestCVSnapshotter(metric_name='loss', asc=True, top=1)
    ]

    return SnapshotConfig(root_dir=config['root'],
                          snapshotters=snapshotters)


def tensorboard_config(config: Dict) -> TensorboardConfig:
    config = config['tracker']['tensorboard']

    return TensorboardConfig(root_dir=config['root'],
                             track_weights=bool(config['track_weights']),
                             track_samples=bool(config['track_samples']),
                             track_predictions=bool(config['track_predictions']))


def dataset_config(config: Dict, name: str) -> DatasetConfig:
    return DatasetConfig(metadata=config[name]['metadata'],
                         annotations=config[name]['annotations'],
                         image_dir=config[name]['image_dir'],
                         lmdb_dir=config[name]['lmdb_dir'])


def dump(df: pd.DataFrame, top: Optional[int] = None, drop_cols: Optional[List[str]] = None) -> str:
    if drop_cols is None:
        drop_cols = []

    df = df.drop(columns=drop_cols)
    if top:
        df = df.head(top)

    return tabulate(df, headers="keys", showindex=False, tablefmt='github')
