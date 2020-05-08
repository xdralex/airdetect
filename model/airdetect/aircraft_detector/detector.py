from dataclasses import dataclass
from typing import Dict
from typing import List

import pytorch_lightning as pl
import torch
import torchvision
from dacite import from_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from wheel5.dataset import TransformDataset
from ..data import coco_categories_unique, reverse_classes, DetectorDatasetConfig, load_detector_dataset


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
        self.categories = coco_categories_unique()
        self.num_categories = len(self.categories)
        self.categories_to_num = reverse_classes(self.categories)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)

        self.model_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self):
        raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError()

    def val_dataloader(self) -> List[DataLoader]:
        raise NotImplementedError()

    @staticmethod
    def load_dataset(config: DetectorDatasetConfig, name: str = ''):
        return load_detector_dataset(config, name)

    def prepare_eval_loader(self, dataset: Dataset, name: str = ''):
        dataset = TransformDataset(dataset, self.model_transform, name=f'{name}-model')

        return DataLoader(dataset,
                          batch_size=self.config.eval_batch,
                          num_workers=self.config.eval_workers,
                          collate_fn=lambda batch: tuple(zip(*batch)),
                          pin_memory=True)

    def configure_optimizers(self):
        raise NotImplementedError()

    def forward(self, x: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        return self.model(x)
