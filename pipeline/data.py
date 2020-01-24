import re
from contextlib import closing
from typing import Dict

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

from numpy.random.mtrand import RandomState


def load_aircraft_data(db_config: Dict, table_name: str = 'aircraft_photos', random_state: RandomState = None) -> pd.DataFrame:
    random_state = random_state or np.random.RandomState()

    # Aircraft type shortening
    airbus_pattern = re.compile(r'Airbus A?(?P<major>\d*)(?P<minor>-\d*)?\s*(\(.*\))?\Z')
    boeing_pattern = re.compile(r'Boeing B?(?P<major>\d*)(?P<minor>-\d*)?\s*(\(.*\))?\Z')
    md_pattern = re.compile(r'McDonnell Douglas MD-(?P<major>\d*)\s*(\(.*\))?\Z')

    def codename(x: str) -> str:
        m = airbus_pattern.match(x)
        if m is not None:
            major = m.group('major')
            minor = '' if m.group('minor') is None else m.group('minor')
            return f'A{major}{minor}'

        m = boeing_pattern.match(x)
        if m is not None:
            major = m.group('major')
            minor = '' if m.group('minor') is None else m.group('minor')
            return f'B{major}{minor}'

        m = md_pattern.match(x)
        if m is not None:
            major = m.group('major')
            return f'MD-{major}'

        raise Exception(f'Unexpected aircraft type <{x}>')

    # Data loading
    conn = psycopg2.connect(**db_config)
    with closing(conn.cursor()) as c:
        c.execute(sql.SQL('''
            SELECT
                (SELECT at.name FROM aircraft_types at WHERE at.id = ap.type_id) AS name,
                ap.path
            FROM {} ap
        ''').format(sql.Identifier(table_name)))

        df_images = pd.DataFrame(c.fetchall(), columns=['name', 'path']).sample(frac=1, random_state=random_state)
    conn.commit()

    df_images['name'] = df_images['name'].map(codename)
    return df_images
