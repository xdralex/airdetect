from typing import NamedTuple, Optional, Callable, List

import pandas as pd
from PIL.Image import Image as Img
from wheel5.dataset import LMDBImageDataset

Transform = Callable[[Img], Img]


class DatasetConfig(NamedTuple):
    metadata: str
    annotations: str
    image_dir: str
    lmdb_dir: str

    classes_path: str


def load_dataset(config: DatasetConfig, target_classes: List[str], store_transform: Optional[Transform] = None) -> LMDBImageDataset:
    df_metadata = pd.read_csv(filepath_or_buffer=config.metadata, sep=',', header=0)
    df_annotations = pd.read_csv(filepath_or_buffer=config.annotations, sep=',', header=0)

    categories_dict = {}
    for row in df_annotations.itertuples():
        categories_dict[row.path] = row.category

    classes_map = {cls: i for i, cls in enumerate(target_classes)}

    df_metadata['target'] = df_metadata['name'].map(lambda name: classes_map[name])
    df_metadata['category'] = df_metadata['path'].map(lambda path: categories_dict[path])

    df_metadata = df_metadata[df_metadata['category'] == 'normal']
    df_metadata = df_metadata.drop(columns=['name', 'category'])

    dataset = LMDBImageDataset.cached(df_metadata,
                                      image_dir=config.image_dir,
                                      lmdb_path=config.lmdb_dir,
                                      transform=store_transform)

    return dataset


def load_classes(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line != '']
