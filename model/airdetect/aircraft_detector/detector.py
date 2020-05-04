import os
from dataclasses import dataclass
from typing import Dict
from typing import List

import albumentations as albu
import cv2
import pandas as pd
import pytorch_lightning as pl
import torchvision
from dacite import from_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from wheel5.dataset import SimpleImageDetectionDataset, AlbumentationsTransform
from wheel5.dataset import TransformDataset
from .data import COCO_INSTANCE_CATEGORY_NAMES


@dataclass
class AircraftDetectorConfig:
    random_state_seed: int

    eval_batch: int = 2
    eval_workers: int = 4


class AircraftDetector(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super(AircraftDetector, self).__init__()

        self.hparams = hparams
        self.config = from_dict(AircraftDetectorConfig, hparams)

        #
        # Model
        #
        na_counter = 0
        self.categories = []
        for category in COCO_INSTANCE_CATEGORY_NAMES:
            if category == 'N/A':
                category = f'{category}_{na_counter}'
                na_counter += 1

            self.categories.append(category)
        self.num_categories = len(self.categories)
        self.categories_to_num = {v: k for k, v in enumerate(self.categories)}
        assert len(self.categories) == len(self.categories_to_num)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)

        # self.initial_transform = AlbumentationsTransform(albu.Compose([
        #     albu.LongestMaxSize(max_size=1024, interpolation=cv2.INTER_AREA),
        # ]))

        self.model_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self):
        raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def val_dataloader(self) -> List[DataLoader]:
        raise NotImplementedError()

    def load_dataset(self, dataset_config: Dict[str, str], name: str = ''):
        image_dir = dataset_config['image_dir']

        metadata = dataset_config.get('metadata')
        if metadata is None:
            entries = []
            for filename in os.listdir(image_dir):
                entries.append({'path': filename})
            df_metadata = pd.DataFrame(entries)
        else:
            raise NotImplementedError()

        return SimpleImageDetectionDataset(df_metadata,
                                           image_dir=image_dir,
                                           name=name)

    def prepare_eval_loader(self, dataset: Dataset, name: str = ''):
        dataset = TransformDataset(dataset, self.model_transform, name=f'{name}-model')

        return DataLoader(dataset,
                          batch_size=self.config.eval_batch,
                          num_workers=self.config.eval_workers,
                          collate_fn=lambda batch: tuple(zip(*batch)),
                          pin_memory=True)

    def configure_optimizers(self):
        raise NotImplementedError()

    def forward(self, x):
        return self.model(x)
