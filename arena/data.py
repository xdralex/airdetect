from typing import Dict, NamedTuple, Optional, Callable, List

import pandas as pd
from PIL.Image import Image as Img
from torch.utils.data import Dataset, DataLoader, SequentialSampler, SubsetRandomSampler
from wheel5.dataset import LMDBImageDataset, SequentialSubsetSampler

Transform = Callable[[Img], Img]


class DataBundle(NamedTuple):
    loader: DataLoader
    dataset: Dataset
    indices: List[int]


def prepare_train_bundle(dataset: Dataset, indices: List[int], batch_size: int = 64, num_workers: int = 4) -> DataBundle:
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

    return DataBundle(loader, dataset, indices)


def prepare_eval_bundle(dataset: Dataset, indices: List[int], randomize: bool, batch_size: int = 256, num_workers: int = 4) -> DataBundle:
    sampler = SubsetRandomSampler(indices) if randomize else SequentialSubsetSampler(indices)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

    return DataBundle(loader, dataset, indices)


def prepare_test_bundle(dataset: Dataset, batch_size: int = 256, num_workers: int = 4) -> DataBundle:
    indices = list(range(len(dataset)))
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)

    return DataBundle(loader, dataset, indices)


def load_dataset(config: Dict[str, str], target_classes: List[str], store_transform: Optional[Transform] = None) -> Dataset:
    df_metadata = pd.read_csv(filepath_or_buffer=config['metadata'], sep=',', header=0)
    df_annotations = pd.read_csv(filepath_or_buffer=config['annotations'], sep=',', header=0)

    categories_dict = {}
    for row in df_annotations.itertuples():
        categories_dict[row.path] = row.category

    classes_map = {cls: i for i, cls in enumerate(target_classes)}

    df_metadata['target'] = df_metadata['name'].map(lambda name: classes_map[name])
    df_metadata['category'] = df_metadata['path'].map(lambda path: categories_dict[path])

    df_metadata = df_metadata[df_metadata['category'] == 'normal']
    df_metadata = df_metadata.drop(columns=['name', 'category'])

    dataset = LMDBImageDataset.cached(df_metadata,
                                      image_dir=config['image_dir'],
                                      lmdb_path=config['lmdb_dir'],
                                      transform=store_transform)

    return dataset
