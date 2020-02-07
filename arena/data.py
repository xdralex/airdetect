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
    df_images = pd.read_csv(filepath_or_buffer=config['dataframe'], sep=',', header=0)

    classes_map = {cls: i for i, cls in enumerate(target_classes)}
    df_images['target'] = df_images['name'].map(lambda x: classes_map[x])
    df_images = df_images.drop(columns='name')

    dataset = LMDBImageDataset.cached(df_images,
                                      image_dir=config['image_dir'],
                                      lmdb_path=config['lmdb_dir'],
                                      transform=store_transform)

    return dataset
