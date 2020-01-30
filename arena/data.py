from typing import Dict, NamedTuple, Optional, Callable

import albumentations as albu
import pandas as pd
from PIL.Image import Image as Img
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
from wheel5.dataset import LMDBImageDataset, TransformDataset, AlbumentationsDataset, DataBundle


class PipelineData(NamedTuple):
    train: DataBundle
    public_test: DataBundle
    private_test: DataBundle


def load_image_dataset(config: Dict[str, str],
                       lmdb_transform: Optional[Callable[[Img], Img]] = None,
                       aug_transform: Optional[albu.BasicTransform] = None,
                       model_transform: Optional[Callable[[Img], Img]] = None) -> Dataset:
    df_images = pd.read_csv(filepath_or_buffer=config['dataframe'], sep=',', header=0)

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
