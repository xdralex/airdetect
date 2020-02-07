from typing import Dict, NamedTuple, Optional, Callable, List

import albumentations as albu
import numpy as np
import pandas as pd
from PIL.Image import Image as Img
from numpy.random.mtrand import RandomState
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
from wheel5.dataset import LMDBImageDataset, TransformDataset, AlbumentationsDataset, split_indices


class DataBundle(NamedTuple):
    loader: DataLoader
    dataset: Dataset
    indices: List[int]


class PipelineData(NamedTuple):
    train: DataBundle
    public_test: DataBundle
    private_test: DataBundle


def target_classes():
    return ['A319',
            'A320',
            'A321',
            'A330-200',
            'A330-300',
            'A340-300',
            'B737-200',
            'B737-300',
            'B737-400',
            'B737-500',
            'B737-700',
            'B737-800',
            'B747-200',
            'B747-400',
            'B757-200',
            'B767-300',
            'B777-200',
            'B777-300',
            'MD-11',
            'MD-80']


def std_images_dataframe(df_images: DataFrame) -> DataFrame:
    classes_map = {cls: i for i, cls in enumerate(target_classes())}

    df = df_images.copy()
    df['target'] = df['name'].map(lambda x: classes_map[x])
    df.drop(columns='name')

    return df


def load_image_dataset(config: Dict[str, str],
                       lmdb_transform: Optional[Callable[[Img], Img]] = None,
                       aug_transform: Optional[albu.BasicTransform] = None,
                       model_transform: Optional[Callable[[Img], Img]] = None) -> Dataset:
    df_images = pd.read_csv(filepath_or_buffer=config['dataframe'], sep=',', header=0)
    df_images = std_images_dataframe(df_images)

    dataset = LMDBImageDataset.cached(df_images,
                                      image_dir=config['image_dir'],
                                      lmdb_path=config['lmdb_dir'],
                                      transform=lmdb_transform)

    if aug_transform:
        dataset = AlbumentationsDataset(dataset, aug_transform)

    if model_transform:
        dataset = TransformDataset(dataset, model_transform)

    return dataset


def load_data(datasets_config: Dict,
              grad_batch: int = 64,
              nograd_batch: int = 256,
              lmdb_transform: Optional[Callable[[Img], Img]] = None,
              aug_transform: Optional[albu.BasicTransform] = None,
              model_transform: Optional[Callable[[Img], Img]] = None) -> PipelineData:
    train_dataset = load_image_dataset(datasets_config['train'], lmdb_transform=lmdb_transform, aug_transform=aug_transform, model_transform=model_transform)
    public_test_dataset = load_image_dataset(datasets_config['public_test'], lmdb_transform=lmdb_transform, model_transform=model_transform)
    private_test_dataset = load_image_dataset(datasets_config['private_test'], lmdb_transform=lmdb_transform, model_transform=model_transform)

    train_indices = list(range(len(train_dataset)))
    public_test_indices = list(range(len(public_test_dataset)))
    private_test_indices = list(range(len(private_test_dataset)))

    train_sampler = SubsetRandomSampler(train_indices)
    public_test_sampler = SubsetRandomSampler(public_test_indices)
    private_test_sampler = SubsetRandomSampler(private_test_indices)

    train_loader = DataLoader(train_dataset, batch_size=grad_batch, sampler=train_sampler, num_workers=4, pin_memory=True)
    public_test_loader = DataLoader(public_test_dataset, batch_size=nograd_batch, sampler=public_test_sampler, num_workers=4, pin_memory=True)
    private_test_loader = DataLoader(private_test_dataset, batch_size=nograd_batch, sampler=private_test_sampler, num_workers=4, pin_memory=True)

    return PipelineData(train=DataBundle(loader=train_loader, dataset=train_dataset, indices=train_indices),
                        public_test=DataBundle(loader=public_test_loader, dataset=public_test_dataset, indices=public_test_indices),
                        private_test=DataBundle(loader=private_test_loader, dataset=private_test_dataset, indices=private_test_indices))


def split_eval_main_data(bundle: DataBundle,
                         split: float,
                         eval_batch: int = 256,
                         main_batch: int = 64,
                         eval_workers: int = 4,
                         main_workers: int = 4,
                         random_state: Optional[RandomState] = None) -> (DataBundle, DataBundle):
    if random_state is None:
        random_state = np.random.RandomState()

    eval_indices, main_indices = split_indices(bundle.indices, split=split, random_state=random_state)

    eval_sampler = SubsetRandomSampler(eval_indices)
    main_sampler = SubsetRandomSampler(main_indices)

    eval_loader = DataLoader(bundle.dataset, batch_size=eval_batch, sampler=eval_sampler, num_workers=eval_workers, pin_memory=True)
    main_loader = DataLoader(bundle.dataset, batch_size=main_batch, sampler=main_sampler, num_workers=main_workers, pin_memory=True)

    eval_bundle = DataBundle(loader=eval_loader, dataset=bundle.dataset, indices=eval_indices)
    main_bundle = DataBundle(loader=main_loader, dataset=bundle.dataset, indices=main_indices)

    return eval_bundle, main_bundle
