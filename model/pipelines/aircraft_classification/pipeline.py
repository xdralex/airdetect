import os
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union
from typing import List, Callable

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from PIL.Image import Image as Img
from dacite import from_dict
from matplotlib.figure import Figure
from numpy.random.mtrand import RandomState
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# noinspection PyProtectedMember
from pytorch_lightning.loggers import rank_zero_only
from torch import nn
from torch.nn import Parameter, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import wheel5.transforms.albumentations as albu_ext
import wheel5.transforms.torchvision as torchviz_ext
from wheel5.dataset import TransformDataset, AlbumentationsDataset, ImageOneHotDataset, ImageCutMixDataset, ImageMixupDataset, targets, LMDBImageDataset
from wheel5.dataset.functional import class_distribution
from wheel5.loss import SoftLabelCrossEntropyLoss
from wheel5.metering import ReservoirSamplingMeter
from wheel5.metrics import ExactMatchAccuracy, JaccardAccuracy
from wheel5.nn import init_softmax_logits, ParamGroup
from wheel5.scheduler import WarmupScheduler
from wheel5.tracking import ProbesInterface
from wheel5.tracking import Tracker, TensorboardLogging, StatisticsTracking, CheckpointPattern

Transform = Callable[[Img], Img]


@dataclass
class AircraftClassificationConfig:
    random_state_seed: int

    classes_path: str
    dataset_config: Dict[str, str]

    repo: str
    network: str

    freeze: int
    mixup: bool
    cutmix: bool

    kv: Dict[str, float]

    logging_sampling_full: bool = False
    logging_samples: int = 8

    val_split: float = 0.2
    train_sample: float = 0.25

    train_batch: int = 32
    train_workers: int = 4
    eval_batch: int = 256
    eval_workers: int = 4


class AircraftClassificationPipeline(pl.LightningModule, ProbesInterface):
    @dataclass
    class Sample:
        x: torch.Tensor
        y: torch.Tensor

    def __init__(self, hparams: Dict):
        super(AircraftClassificationPipeline, self).__init__()

        self.hparams = hparams
        self.config = from_dict(AircraftClassificationConfig, hparams)

        self.random_state = RandomState(self.config.random_state_seed)

        self.target_classes = self._load_classes(self.config.classes_path)
        self.num_classes = len(self.target_classes)

        self.train_loader = None
        self.train_subset_loader = None
        self.train_orig_loader = None
        self.val_loader = None

        self.epoch_fit_metrics = None
        self.epoch_samples: Dict[str, ReservoirSamplingMeter] = {}
        self.sample_transform = None

        #
        # Model
        #
        self.model = torch.hub.load(self.config.repo, self.config.network, pretrained=True, verbose=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)
        self.group_names, self.optimizer_params = self.adjust_model_params()

        self.train_accuracy = JaccardAccuracy()
        self.train_loss = SoftLabelCrossEntropyLoss(
            smooth_factor=self.config.kv['lb_smooth'],
            smooth_dist=torch.full([self.num_classes], fill_value=1.0 / self.num_classes))

        self.eval_accuracy = ExactMatchAccuracy()
        self.eval_loss = CrossEntropyLoss()

        #
        # Transforms
        #
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        mean_color = tuple([int(round(c * 255)) for c in normalize_mean])

        self.store_transform = transforms.Compose([
            torchviz_ext.Rescale(scale=0.5, interpolation=Image.LANCZOS)
        ])

        self.train_transform_pre_cutmix = albu.Compose([
            albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1.0),
            albu.HorizontalFlip(p=0.5)
        ])

        self.train_transform_pre_mixup = albu.Compose([
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

        self.train_transform_final = albu.Compose([
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

        self.eval_transform = albu.Compose([
            albu_ext.PadToSquare(fill=mean_color),
            albu_ext.Resize(height=224, width=224, interpolation=cv2.INTER_AREA)
        ])

        self.model_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

        self.sample_transform = torchviz_ext.InvNormalize(mean=normalize_mean, std=normalize_std)

    def adjust_model_params(self):
        param_groups = {
            'A': ParamGroup({'lr': self.config.kv['lrA'], 'weight_decay': self.config.kv['wdA']}),
            'A_no_decay': ParamGroup({'lr': self.config.kv['lrA'], 'weight_decay': 0}),
            'B': ParamGroup({'lr': self.config.kv['lrB'], 'weight_decay': self.config.kv['wdB']}),
            'B_no_decay': ParamGroup({'lr': self.config.kv['lrB'], 'weight_decay': 0})
        }

        def add_param(group_name: str, module_name: str, param_name: str, param: Parameter):
            if param.requires_grad:
                group = param_groups[group_name]
                group.params.append((f'{module_name}.{param_name}', param))

        def freeze_params():
            for index, (name, child) in enumerate(self.model.named_children()):
                if index < self.config.freeze:
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
        return prepare_params()

    def prepare_data(self):
        #
        # Dataset loading
        #
        fit_dataset = self.load_dataset(self.config.dataset_config)

        # TODO: shuffle loader?

        #
        # Split into train/val/ctrl
        #
        fit_indices = list(range(0, len(fit_dataset)))
        self.random_state.shuffle(fit_indices)

        val_divider = int(np.round(self.config.val_split * len(fit_indices)))
        before_val_indices, after_val_indices = fit_indices[val_divider:], fit_indices[:val_divider]

        train_root_dataset = Subset(fit_dataset, before_val_indices)
        val_dataset = Subset(fit_dataset, after_val_indices)

        train_targets = targets(train_root_dataset)

        train_root_indices = list(range(0, len(train_targets)))
        train_sample_size = int(len(train_root_indices) * self.config.train_sample)
        train_sample_indices = self.random_state.choice(train_root_indices, size=train_sample_size, replace=False)

        #
        # Training
        #
        train_dataset = ImageOneHotDataset(train_root_dataset, self.num_classes)

        train_dataset = AlbumentationsDataset(train_dataset, self.train_transform_pre_cutmix)
        if self.config.cutmix:
            cutmix_alpha = self.config.kv['cutmix_alpha']
            train_dataset = ImageCutMixDataset(train_dataset, alpha=cutmix_alpha, random_state=self.random_state)

        train_dataset = AlbumentationsDataset(train_dataset, self.train_transform_pre_mixup)
        if self.config.mixup:
            mixup_alpha = self.config.kv['mixup_alpha']
            train_dataset = ImageMixupDataset(train_dataset, alpha=mixup_alpha, random_state=self.random_state)

        train_dataset = AlbumentationsDataset(train_dataset, self.train_transform_final)
        train_dataset = TransformDataset(train_dataset, self.model_transform)
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.config.train_batch,
                                       num_workers=self.config.train_workers,
                                       pin_memory=True)

        #
        # Validation
        #
        self.train_subset_loader = DataLoader(Subset(train_dataset, train_sample_indices),
                                              batch_size=self.config.eval_batch,
                                              num_workers=self.config.eval_workers,
                                              pin_memory=True)

        self.train_orig_loader = self.prepare_eval_loader(Subset(train_root_dataset, train_sample_indices))
        self.val_loader = self.prepare_eval_loader(val_dataset)

        #
        # Model adjustment
        #
        target_probs = class_distribution(train_targets, self.num_classes)
        init_softmax_logits(self.model.fc.bias, torch.from_numpy(target_probs))

    def load_dataset(self, dataset_config: Dict[str, str]):
        return self._load_dataset(dataset_config, self.target_classes, self.store_transform)

    def prepare_eval_loader(self, dataset: Dataset):
        dataset = AlbumentationsDataset(dataset, self.eval_transform)
        dataset = TransformDataset(dataset, self.model_transform)

        return DataLoader(dataset,
                          batch_size=self.config.eval_batch,
                          num_workers=self.config.eval_workers,
                          pin_memory=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.optimizer_params)

        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=int(round(self.config.kv['lr_t0'])),
                                                T_mult=int(round(self.config.kv['lr_f'])))
        scheduler = WarmupScheduler(optimizer, epochs=self.config.kv['lr_warmup'], next_scheduler=scheduler)

        return [optimizer], [scheduler]

    def on_epoch_start(self):
        self.epoch_fit_metrics = None
        self.epoch_samples = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Dict:
        x, y, _ = batch

        z = self.forward(x)

        loss = self.train_loss(z, y)
        self.add_epoch_samples('train', batch_idx, x, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int) -> Dict:
        x, y, _ = batch

        z = self.forward(x)
        y_probs = torch.exp(log_softmax(z, dim=1))
        y_hat = torch.argmax(y_probs, dim=1)

        if dataloader_idx == 0:  # val_loader
            loss = self.eval_loss(z, y)
            correct, total = self.eval_accuracy(y_hat, y)
            prefix = 'val'
        elif dataloader_idx == 1:  # train_subset_loader
            loss = self.train_loss(z, y)
            correct, total = self.train_accuracy(y_probs, y)
            prefix = 'train-aug'
        elif dataloader_idx == 2:  # train_orig_loader
            loss = self.eval_loss(z, y)
            correct, total = self.eval_accuracy(y_hat, y)
            prefix = 'train-orig'
        else:
            raise AssertionError(f'Invalid dataloader index: {dataloader_idx}')

        self.add_epoch_samples(prefix, batch_idx, x, y)

        return {
            f'{prefix}_loss': loss,
            f'{prefix}_correct': correct,
            f'{prefix}_total': total
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        metrics = OrderedDict()
        for dataloader_idx, output in enumerate(outputs):
            if dataloader_idx == 0:  # val_loader
                prefix = 'val'
            elif dataloader_idx == 1:  # train_subset_loader
                prefix = 'train-aug'
            elif dataloader_idx == 2:  # train_orig_loader
                prefix = 'train-orig'
            else:
                raise AssertionError(f'Invalid dataloader index: {dataloader_idx}')

            loss = torch.stack([x[f'{prefix}_loss'] for x in output]).mean()

            correct_sum = torch.stack([x[f'{prefix}_correct'] for x in output]).sum()
            total_sum = torch.stack([x[f'{prefix}_total'] for x in output]).sum()
            accuracy = correct_sum / float(total_sum)

            metrics[f'{prefix}_loss'] = loss
            metrics[f'{prefix}_acc'] = accuracy

        self.epoch_fit_metrics = metrics
        progress_bar = metrics

        val_acc = metrics['val_acc']
        val_loss = metrics['val_acc']

        return {
            'val_acc': val_acc,
            'val_loss': val_loss,
            'progress_bar': progress_bar,
        }

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> List[DataLoader]:
        return [self.val_loader, self.train_subset_loader, self.train_orig_loader]

    def add_epoch_samples(self, name: str, batch_idx: int, x: torch.Tensor, y: torch.Tensor):
        if batch_idx == 0 or self.config.logging_sampling_full:
            if name not in self.epoch_samples:
                self.epoch_samples[name] = ReservoirSamplingMeter(k=self.config.logging_samples)

            elements = []
            for i in range(0, y.shape[0]):
                sample = self.Sample(x=x[i], y=y[i])
                elements.append(sample)
            self.epoch_samples[name].add(elements)

    def probe_epoch_fit_metrics(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.epoch_fit_metrics.items()}

    def probe_epoch_aux_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        optimizer = self.trainer.optimizers[0]
        assert len(self.group_names) == len(optimizer.param_groups)

        for group_name, param_group in zip(self.group_names, optimizer.param_groups):
            for k, v in param_group.items():
                if k == 'lr':
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        v = None

                    if v is not None:
                        metrics[f'optim/{group_name}/{k}'] = v

        return metrics

    def probe_epoch_histograms(self) -> Dict[str, torch.Tensor]:
        return {f'weights/{k}': v for k, v in self.named_parameters()}

    def probe_epoch_images(self) -> Dict[str, List[torch.Tensor]]:
        def meter_to_tensors(meter: ReservoirSamplingMeter) -> List[torch.Tensor]:
            return [self.sample_transform(sample.x) for sample in meter.value()]

        return {f'samples/{name}': meter_to_tensors(meter) for name, meter in self.epoch_samples.items()}

    def probe_epoch_figures(self) -> Dict[str, Figure]:
        #     if prediction is not None:
        #         cm_fig = visualize_cm(classes, y_true=prediction['y'].numpy(), y_pred=prediction['y_hat'].numpy())
        #         self.tb_writer.add_figure(f'predictions/cm', cm_fig, state.epoch)
        #
        #         pil_image_transform = ToPILImage()
        #
        #         def viz_transform(x):
        #             return pil_image_transform(self.sample_img_transform(x))
        #
        #         mismatch_figs = visualize_top_errors(classes,
        #                                              y_true=prediction['y'].numpy(),
        #                                              y_pred=prediction['y_hat'].numpy(),
        #                                              image_indices=prediction['indices'].numpy(),
        #                                              image_dataset=TransformDataset(prediction_dataset, viz_transform))
        return {}

    def get_tqdm_dict(self):
        d = dict(super(AircraftClassificationPipeline, self).get_tqdm_dict())
        del d['v_num']
        return d

    @staticmethod
    def _load_dataset(config: Dict[str, str], target_classes: List[str], store_transform: Optional[Transform] = None) -> LMDBImageDataset:
        metadata = config['metadata']
        annotations = config['annotations']
        image_dir = config['image_dir']
        lmdb_dir = config['lmdb_dir']

        df_metadata = pd.read_csv(filepath_or_buffer=metadata, sep=',', header=0)
        df_annotations = pd.read_csv(filepath_or_buffer=annotations, sep=',', header=0)

        categories_dict = {}
        for row in df_annotations.itertuples():
            categories_dict[row.path] = row.category

        classes_map = {cls: i for i, cls in enumerate(target_classes)}

        df_metadata['target'] = df_metadata['name'].map(lambda name: classes_map[name])
        df_metadata['category'] = df_metadata['path'].map(lambda path: categories_dict[path])

        df_metadata = df_metadata[df_metadata['category'] == 'normal']
        df_metadata = df_metadata.drop(columns=['name', 'category'])

        dataset = LMDBImageDataset.cached(df_metadata,
                                          image_dir=image_dir,
                                          lmdb_path=lmdb_dir,
                                          transform=store_transform)

        return dataset

    @staticmethod
    def _load_classes(path: str) -> List[str]:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            return [line for line in lines if line != '']


class TensorboardHparamsLogger(TensorBoardLogger):
    def __init__(self, save_dir: str, name: Optional[str] = "default", version: Optional[Union[int, str]] = None, **kwargs):
        super(TensorboardHparamsLogger, self).__init__(save_dir, name, version, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params: Dict) -> None:
        config = from_dict(AircraftClassificationConfig, params)
        super(TensorboardHparamsLogger, self).log_hyperparams(config.kv)


def eval_blend(dataset_config: Dict[str, str],
               device: int,
               snapshot_paths: List[str],
               show_progress: bool = True) -> Dict[str, float]:

    with torch.no_grad():
        loader = None

        y_only = None
        y_probs_list = []
        for i, snapshot_path in enumerate(snapshot_paths):
            model = AircraftClassificationPipeline.load_from_checkpoint(snapshot_path, map_location=torch.device(f'cuda:{device}'))
            model.freeze()

            if loader is None:
                dataset = model.load_dataset(dataset_config)
                loader = model.prepare_eval_loader(dataset)

            z_list = []
            y_list = []
            with tqdm(total=len(loader), disable=not show_progress) as progress_bar:
                progress_bar.set_description(f'Evaluating model {i + 1}')

                for x, y, _ in loader:
                    z_list.append(model.forward(x))
                    y_list.append(y)

                    progress_bar.update()

            z = torch.cat(z_list)
            y = torch.cat(y_list)
            y_probs = torch.exp(log_softmax(z, dim=1))

            y_probs_list.append(y_probs)
            if y_only is None:
                y_only = y
            del model

        y_probs_stack = torch.stack(y_probs_list, dim=0)
        y_probs_blend = torch.mean(y_probs_stack, dim=0)
        y_hat = torch.argmax(y_probs_blend, dim=1)

        model = AircraftClassificationPipeline.load_from_checkpoint(snapshot_paths[0], map_location=torch.device(f'cuda:{device}'))

        correct, total = model.eval_accuracy(y_hat, y_only)
        accuracy = float(correct) / float(total)

        return {
            'acc': accuracy
        }


def fit_trial(tracker: Tracker,
              snapshot_dir: str,
              tensorboard_root: str,
              experiment: str,
              device: int,
              config: AircraftClassificationConfig,
              max_epochs: int) -> Dict[str, float]:

    trial_tracker = tracker.new_trial(config.kv)
    snapshot_trial = os.path.join(snapshot_dir, trial_tracker.trial)

    pipeline = AircraftClassificationPipeline(hparams=asdict(config))
    logger = TensorboardHparamsLogger(save_dir=tensorboard_root, name=experiment, version=trial_tracker.trial)

    checkpoint_callback = ModelCheckpoint(filepath=CheckpointPattern.pattern(snapshot_trial), monitor='val_acc', mode='max', save_top_k=1)
    early_stop_callback = False
    tensorboard_callback = TensorboardLogging()
    tracking_callback = StatisticsTracking(trial_tracker)

    trainer = Trainer(logger=logger,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback,
                      callbacks=[tensorboard_callback, tracking_callback],
                      gpus=[device],
                      max_epochs=max_epochs,
                      progress_bar_refresh_rate=1,
                      num_sanity_val_steps=0)

    trainer.fit(pipeline)

    metrics_df = tracking_callback.metrics_df()

    return {
        'min_val_loss': metrics_df['val_loss'].min(),
        'max_val_acc': metrics_df['val_acc'].max(),
        'min_train_loss': metrics_df['train_loss'].min(),
        'max_train_acc': metrics_df['train_acc'].max(),
        'min_train-aug_loss': metrics_df['train-aug_loss'].min(),
        'max_train-aug_acc': metrics_df['train-aug_acc'].max()
    }
