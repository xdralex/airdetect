from dataclasses import dataclass
from typing import Dict, Union, Tuple, List, cast

import albumentations as albu
import cv2
import numpy as np
import torch
import wheel5.transforms.albumentations as albu_ext
import wheel5.transforms.torchvision as torchviz_ext
from PIL import Image
from numpy.random.mtrand import RandomState
from torch import nn
from torch.nn import Parameter, CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader
from torch.nn.functional import log_softmax
import pytorch_lightning as pl
from torchvision import transforms
from wheel5.dataset import TransformDataset, AlbumentationsDataset, LMDBImageDataset, ImageOneHotDataset, ImageCutMixDataset, ImageMixupDataset, \
    SequentialSubsetSampler, targets
from wheel5.dataset.functional import class_distribution
from wheel5.loss import SoftLabelCrossEntropyLoss
from wheel5.metrics import ExactMatchAccuracy, JaccardAccuracy
from wheel5.model import fit
from wheel5.model import score_blend
from wheel5.nn import init_softmax_logits, ParamGroup
from wheel5.scheduler import WarmupScheduler
from wheel5.tracking import Tracker, Snapshotter

from data import load_dataset, DatasetConfig, load_classes


@dataclass
class DatasetConfig:
    metadata: str
    annotations: str
    image_dir: str
    lmdb_dir: str


@dataclass
class AircraftClassificationConfig:
    random_state_seed: int

    classes_path: str
    fit_dataset_config: DatasetConfig
    test_dataset_config: DatasetConfig

    repo: str
    network: str

    freeze: int
    mixup: bool
    cutmix: bool

    hparams: Dict[str, float]

    print_model_transforms: bool = True

    val_split: float = 0.2

    train_batch: int = 32
    train_workers: int = 4
    eval_batch: int = 256
    eval_workers: int = 4


class AircraftClassificationPipeline(pl.LightningModule):

    def __init__(self, config: AircraftClassificationConfig):
        super(AircraftClassificationPipeline, self).__init__()

        self.config = config
        self.random_state = RandomState(config.random_state_seed)

        self.target_classes = load_classes(config.classes_path)
        self.num_classes = len(self.target_classes)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.model = torch.hub.load(self.config.repo, self.config.network, pretrained=True, verbose=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)

        self.group_names, self.optimizer_params = self.adjust_model_params()

        smooth_dist = torch.full([self.num_classes], fill_value=1.0 / self.num_classes)
        self.train_loss = SoftLabelCrossEntropyLoss(smooth_factor=self.config.hparams['loss_smooth'], smooth_dist=smooth_dist)
        self.train_accuracy = JaccardAccuracy()

        self.eval_loss = CrossEntropyLoss()
        self.eval_accuracy = ExactMatchAccuracy()

    def adjust_model_params(self):
        param_groups = {
            'A': ParamGroup({'lr': self.config.hparams['lrA'], 'weight_decay': self.config.hparams['wdA']}),
            'A_no_decay': ParamGroup({'lr': self.config.hparams['lrA'], 'weight_decay': 0}),
            'B': ParamGroup({'lr': self.config.hparams['lrB'], 'weight_decay': self.config.hparams['wdB']}),
            'B_no_decay': ParamGroup({'lr': self.config.hparams['lrB'], 'weight_decay': 0})
        }

        def add_param(group_name: str, module_name: str, param_name: str, param: Parameter):
            if param.requires_grad:
                group = param_groups[group_name]
                group.params.append((f'{module_name}.{param_name}', param))

        def freeze_params():
            for index, (name, child) in enumerate(self.model.named_children()):
                if index < self.config.freeze:
                    if self.config.print_model_transforms:
                        print(f'Freezing layer {name}')

                    for param in child.parameters(recurse=True):
                        param.requires_grad = False

        def init_param_groups():
            for module_name, module in self.model.named_modules():
                if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
                    for param_name, param in module.named_parameters(recurse=False):
                        add_param('A_no_decay', module_name, param_name, param)

                elif module_name == 'fc':
                    for param_name, param in module.named_parameters(recurse=False):
                        group = 'B_no_decay' if param_name == 'bias' else 'B'
                        add_param(group, module_name, param_name, param)

                else:
                    for param_name, param in module.named_parameters(recurse=False):
                        group = 'A_no_decay' if param_name == 'bias' else 'A'
                        add_param(group, module_name, param_name, param)

        def print_param_groups():
            if self.config.print_model_transforms:
                for group_name, group in param_groups.items():
                    print(f'{group_name}: {group}')

        def prepare_params():
            group_names = []
            optimizer_params = []
            for group_name, group in param_groups.items():
                if len(group.params) > 0:
                    entry = {'params': group.parameters()}
                    entry.update(group.config)

                    group_names.append(group_name)
                    optimizer_params.append(entry)

            return group_names, optimizer_params

        freeze_params()
        init_param_groups()
        print_param_groups()

        return prepare_params()

    def prepare_data(self):
        #
        # Transforms
        #
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        mean_color = tuple([int(round(c * 255)) for c in normalize_mean])

        store_transform = transforms.Compose([
            torchviz_ext.Rescale(scale=0.5, interpolation=Image.LANCZOS)
        ])

        train_transform_pre_cutmix = albu.Compose([
            albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            albu.HorizontalFlip(p=0.5)
        ])

        train_transform_pre_mixup = albu.Compose([
            albu_ext.PadToSquare(fill=mean_color),
            albu.ShiftScaleRotate(shift_limit=0.1,
                                  scale_limit=(-0.25, 0.15),
                                  rotate_limit=20,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  value=mean_color,
                                  interpolation=cv2.INTER_AREA,
                                  p=1.0),
            albu_ext.Resize(height=224, width=224, interpolation=cv2.INTER_AREA),
        ])

        train_transform_final = albu.Compose([
            albu.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            albu.CoarseDropout(max_holes=12,
                               max_height=12,
                               max_width=12,
                               min_holes=6,
                               min_height=6,
                               min_width=6,
                               fill_value=mean_color,
                               p=1.0)
        ])

        eval_transform = albu.Compose([
            albu_ext.PadToSquare(fill=mean_color),
            albu_ext.Resize(height=224, width=224, interpolation=cv2.INTER_AREA)
        ])

        model_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

        #
        # Dataset loading
        #
        fit_dataset = load_dataset(self.config.fit_dataset_config, self.target_classes, store_transform)
        test_dataset = load_dataset(self.config.test_dataset_config, self.target_classes, store_transform)

        #
        # Split into train/val/ctrl
        #
        fit_indices = list(range(0, len(fit_dataset)))
        self.random_state.shuffle(fit_indices)

        val_divider = int(np.round(self.config.val_split * len(fit_indices)))
        train_indices, val_indices = fit_indices[val_divider:], fit_indices[:val_divider]

        train_dataset = Subset(fit_dataset, train_indices)
        val_dataset = Subset(fit_dataset, val_indices)

        train_dataset_targets = targets(train_dataset)

        #
        # Train transformations
        #
        train_dataset = ImageOneHotDataset(train_dataset, self.num_classes)

        train_dataset = AlbumentationsDataset(train_dataset, train_transform_pre_cutmix)
        if self.config.cutmix:
            cutmix_alpha = self.config.hparams['cutmix_alpha']
            train_dataset = ImageCutMixDataset(train_dataset, alpha=cutmix_alpha, random_state=self.random_state)

        train_dataset = AlbumentationsDataset(train_dataset, train_transform_pre_mixup)
        if self.config.mixup:
            mixup_alpha = self.config.hparams['mixup_alpha']
            train_dataset = ImageMixupDataset(train_dataset, alpha=mixup_alpha, random_state=self.random_state)

        train_dataset = AlbumentationsDataset(train_dataset, train_transform_final)
        train_dataset = TransformDataset(train_dataset, model_transform)

        #
        # Eval transformations
        #
        val_dataset = AlbumentationsDataset(val_dataset, eval_transform)
        val_dataset = TransformDataset(val_dataset, model_transform)

        test_dataset = AlbumentationsDataset(test_dataset, eval_transform)
        test_dataset = TransformDataset(test_dataset, model_transform)

        #
        # Data loading
        #
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.config.train_batch,
                                       num_workers=self.config.train_workers,
                                       pin_memory=True)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.config.eval_batch,
                                     num_workers=self.config.eval_workers,
                                     pin_memory=True)

        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.config.eval_batch,
                                      num_workers=self.config.eval_workers,
                                      pin_memory=True)

        #
        # Model adjustment
        #
        target_probs = class_distribution(train_dataset_targets, self.num_classes)
        init_softmax_logits(self.model.fc.bias, torch.from_numpy(target_probs))

    def configure_optimizers(self):
        optimizer = AdamW(self.optimizer_params)

        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=int(round(self.config.hparams['lr_cos_t0'])),
                                                T_mult=int(round(self.config.hparams['lr_cos_f'])))
        scheduler = WarmupScheduler(optimizer, epochs=self.config.hparams['lr_warmup_epochs'], next_scheduler=scheduler)

        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Dict:
        x, y, _ = batch

        z = self.forward(x)

        loss = self.train_loss(z, y)
        return {'loss': loss}

    def training_epoch_end(self, outputs) -> Dict:
        pass

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def validation_step(self, batch, batch_idx) -> Dict:
        x, y, _ = batch

        z = self.forward(x)
        y_probs = torch.exp(log_softmax(z, dim=1))
        y_hat = torch.argmax(y_probs, dim=1)

        loss = self.eval_loss(z, y)
        correct, total = self.eval_accuracy(y_hat, y)
        return {'val_loss': loss, 'val_correct': correct, 'val_total': total}

    def validation_epoch_end(self, outputs) -> Dict:
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_correct_sum = torch.stack([x['val_correct'] for x in outputs]).sum()
        val_total_sum = torch.stack([x['val_total'] for x in outputs]).sum()
        val_accuracy = val_correct_sum / float(val_total_sum)

        return {'progress_bar': {'val_loss': val_loss, 'val_acc': val_accuracy}}

    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    def test_step(self, batch, batch_idx) -> Dict:
        x, y, _ = batch

        y_hat = self.forward(x)
        loss = self.eval_loss(y_hat, y)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs) -> Dict:
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': val_loss_mean}

    def test_dataloader(self) -> DataLoader:
        return self.test_loader
