import os
from dataclasses import dataclass
from typing import Dict
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torchvision
from dacite import from_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from airdetect.aircraft_detection.data import COCO_INSTANCE_CATEGORY_NAMES
from wheel5.dataset import SimpleImageDetectionDataset
from wheel5.dataset import TransformDataset


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
        self.categories = COCO_INSTANCE_CATEGORY_NAMES
        self.num_categories = len(self.categories)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)

        #
        # Transforms
        #
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        self.model_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
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
            # TODO
            raise NotImplementedError()

        return SimpleImageDetectionDataset(df_metadata,
                                           image_dir=image_dir,
                                           name=name)

    def prepare_eval_loader(self, dataset: Dataset, name: str = ''):
        dataset = TransformDataset(dataset, self.model_transform, name=f'{name}-model')

        return DataLoader(dataset,
                          batch_size=self.config.eval_batch,
                          num_workers=self.config.eval_workers,
                          pin_memory=True)

    def configure_optimizers(self):
        raise NotImplementedError()

    def forward(self, x):
        return self.model(x)
