import logging
from typing import Dict, List, Optional

import pandas as pd
from tabulate import tabulate
from tensorboard import program
from wheel5.tracking import CheckpointSnapshotter, BestCVSnapshotter, SnapshotConfig, TensorboardConfig


def launch_tensorboard(tensorboard_root: str, port: int = 6006):
    logger = logging.getLogger('arena.util')

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--bind_all', '--port', f'{port}', '--logdir', tensorboard_root])
    url = tb.launch()

    logger.info(f'Launched TensorBoard at {url}')


def snapshot_config(config: Dict) -> SnapshotConfig:
    return SnapshotConfig(root_dir=config['tracker']['snapshot_root'],
                          snapshotters=[
                              CheckpointSnapshotter(frequency=20),
                              BestCVSnapshotter(metric_name='acc', asc=False, top=5),
                              BestCVSnapshotter(metric_name='loss', asc=True, top=5)])


def tensorboard_config(config: Dict) -> TensorboardConfig:
    return TensorboardConfig(root_dir=config['tracker']['tensorboard_root'])


def dump(df: pd.DataFrame, top: Optional[int] = None, drop_cols: Optional[List[str]] = None) -> str:
    if drop_cols is None:
        drop_cols = ['experiment', 'trial', 'time', 'directory']

    df = df.drop(columns=drop_cols)
    if top:
        df = df.head(top)

    return tabulate(df, headers="keys", showindex=False, tablefmt='github')
