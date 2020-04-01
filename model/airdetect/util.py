import logging
import math
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


def dump(df: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> str:
    def format_float(v: float) -> str:
        if abs(int(v) - v) < 1e-6:
            return f'{v:.1f}'

        if abs(v) < 1e-3 or abs(v) >= 1e+5:
            return f'{v:.2e}'

        zeros = math.ceil(math.log10(math.fabs(v) + 1))
        if zeros < 5:
            return f'{v:.{5 - zeros}f}'
        else:
            return f'{v:.1f}'

    if drop_cols is None:
        drop_cols = []

    df = df.drop(columns=drop_cols)
    for col in list(df.columns):
        df[col] = df[col].apply(lambda x: format_float(x) if isinstance(x, float) else str(x))

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
