import logging
from collections import OrderedDict
from typing import List, Optional, Dict, Tuple

import pandas as pd
from tabulate import tabulate
from tensorboard import program
import click


def launch_tensorboard(tensorboard_root: str, port: int = 6006):
    logger = logging.getLogger(__name__)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--bind_all', '--port', f'{port}', '--logdir', tensorboard_root])
    url = tb.launch()

    logger.info(f'Launched TensorBoard at {url}')


def dump(df: pd.DataFrame, top: Optional[int] = None, drop_cols: Optional[List[str]] = None) -> str:
    if drop_cols is None:
        drop_cols = []

    df = df.drop(columns=drop_cols)
    if top:
        df = df.head(top)

    return tabulate(df, headers="keys", showindex=False, tablefmt='github')


def parse_kv(kv: str) -> Dict[str, float]:
    def parse_kv_entry(entry: str) -> Tuple[str, float]:
        tokens = [token.strip() for token in entry.split('=') if token.strip() != '']
        if len(tokens) != 2:
            raise click.BadOptionUsage('kv', f'Invalid key-value entry: "{entry}"')

        k, v = tuple(tokens)
        try:
            return k, float(v)
        except (ValueError, TypeError):
            raise click.BadOptionUsage('kv', f'Invalid key-value entry: "{entry}"')

    kv_entries = [entry.strip() for entry in kv.split(',') if entry.strip() != '']
    kv_dict = OrderedDict()

    for entry in kv_entries:
        entry_k, entry_v = parse_kv_entry(entry)
        kv_dict[entry_k] = entry_v

    return kv_dict
