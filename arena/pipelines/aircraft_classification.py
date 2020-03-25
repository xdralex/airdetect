import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional
from typing import List, Callable

import albumentations as albu
import click
import cv2
import hyperopt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from PIL import Image
from PIL.Image import Image as Img
from hyperopt import hp, fmin
from matplotlib.figure import Figure
from numpy.random.mtrand import RandomState
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import Parameter, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

import wheel5.transforms.albumentations as albu_ext
import wheel5.transforms.torchvision as torchviz_ext
from util import launch_tensorboard, parse_hparams
from wheel5 import logutils
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


def make_space_dict() -> Dict[str, Dict]:
    return {
        'resnet50_narrow': {
            'lrA': hp.uniform('lrA', 2e-4, 4e-4),
            'wdA': hp.loguniform('wdA', math.log(1e-2), math.log(1)),
            'lrB': hp.uniform('lrB', 2e-4, 4e-4),
            'wdB': hp.loguniform('wdB', math.log(1e-2), math.log(1)),
            'lr_t0': hp.choice('lr_t0', [10]),
            'lr_f': hp.choice('lr_t0', [2.0]),
            'lr_warmup': hp.choice('lr_warmup', [3]),
            'lb_smooth': hp.loguniform('lb_smooth', math.log(1e-4), math.log(1)),
            'cutmix_alpha': hp.loguniform('cutmix_alpha', math.log(1e-2), math.log(1)),
            'mixup_alpha': hp.loguniform('mixup_alpha', math.log(1e-2), math.log(1))
        },
        'resnet50_wide': {
            'lrA': hp.loguniform('lrA', math.log(1e-4), math.log(1e-2)),
            'wdA': hp.loguniform('wdA', math.log(1e-3), math.log(1e+1)),
            'lrB': hp.loguniform('lrB', math.log(1e-4), math.log(1e-2)),
            'wdB': hp.loguniform('wdB', math.log(1e-3), math.log(1e+1)),
            'lr_t0': hp.choice('lr_t0', [10]),
            'lr_f': hp.choice('lr_t0', [2.0]),
            'lr_warmup': hp.choice('lr_warmup', [3]),
            'lb_smooth': hp.loguniform('smooth', math.log(1e-5), math.log(1)),
            'cutmix_alpha': hp.loguniform('cutmix_alpha', math.log(1e-3), math.log(1e+1)),
            'mixup_alpha': hp.loguniform('mixup_alpha', math.log(1e-3), math.log(1e+1))
        }
    }


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

    print_model_transforms: bool = True
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

    def __init__(self, config: AircraftClassificationConfig, hparams: Dict[str, float]):
        super(AircraftClassificationPipeline, self).__init__()

        self.config = config
        self.hparams = hparams
        self.random_state = RandomState(config.random_state_seed)

        self.target_classes = self.load_classes(config.classes_path)
        self.num_classes = len(self.target_classes)

        self.train_loader = None
        self.train_subset_loader = None
        self.train_orig_loader = None
        self.val_loader = None

        self.model = torch.hub.load(self.config.repo, self.config.network, pretrained=True, verbose=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)

        self.group_names, self.optimizer_params = self.adjust_model_params()

        smooth_dist = torch.full([self.num_classes], fill_value=1.0 / self.num_classes)
        self.train_loss = SoftLabelCrossEntropyLoss(smooth_factor=self.hparams['lb_smooth'], smooth_dist=smooth_dist)
        self.train_accuracy = JaccardAccuracy()

        self.eval_loss = CrossEntropyLoss()
        self.eval_accuracy = ExactMatchAccuracy()

        self.epoch_fit_metrics = None

        self.epoch_samples: Dict[str, ReservoirSamplingMeter] = {}
        self.sample_transform = None

    def adjust_model_params(self):
        param_groups = {
            'A': ParamGroup({'lr': self.hparams['lrA'], 'weight_decay': self.hparams['wdA']}),
            'A_no_decay': ParamGroup({'lr': self.hparams['lrA'], 'weight_decay': 0}),
            'B': ParamGroup({'lr': self.hparams['lrB'], 'weight_decay': self.hparams['wdB']}),
            'B_no_decay': ParamGroup({'lr': self.hparams['lrB'], 'weight_decay': 0})
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

        self.sample_transform = torchviz_ext.InvNormalize(mean=normalize_mean, std=normalize_std)

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
        fit_dataset = self.load_dataset(self.config.dataset_config, self.target_classes, store_transform)

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
        # Train transformations
        #
        train_dataset = ImageOneHotDataset(train_root_dataset, self.num_classes)

        train_dataset = AlbumentationsDataset(train_dataset, train_transform_pre_cutmix)
        if self.config.cutmix:
            cutmix_alpha = self.hparams['cutmix_alpha']
            train_dataset = ImageCutMixDataset(train_dataset, alpha=cutmix_alpha, random_state=self.random_state)

        train_dataset = AlbumentationsDataset(train_dataset, train_transform_pre_mixup)
        if self.config.mixup:
            mixup_alpha = self.hparams['mixup_alpha']
            train_dataset = ImageMixupDataset(train_dataset, alpha=mixup_alpha, random_state=self.random_state)

        train_dataset = AlbumentationsDataset(train_dataset, train_transform_final)
        train_dataset = TransformDataset(train_dataset, model_transform)

        train_subset_dataset = Subset(train_dataset, train_sample_indices)

        #
        # Eval transformations
        #
        train_orig_dataset = Subset(train_root_dataset, train_sample_indices)
        train_orig_dataset = AlbumentationsDataset(train_orig_dataset, eval_transform)
        train_orig_dataset = TransformDataset(train_orig_dataset, model_transform)

        val_dataset = AlbumentationsDataset(val_dataset, eval_transform)
        val_dataset = TransformDataset(val_dataset, model_transform)

        #
        # Data loading
        #
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.config.train_batch,
                                       num_workers=self.config.train_workers,
                                       pin_memory=True)

        self.train_subset_loader = DataLoader(train_subset_dataset,
                                              batch_size=self.config.eval_batch,
                                              num_workers=self.config.eval_workers,
                                              pin_memory=True)

        self.train_orig_loader = DataLoader(train_orig_dataset,
                                            batch_size=self.config.eval_batch,
                                            num_workers=self.config.eval_workers,
                                            pin_memory=True)

        self.val_loader = DataLoader(val_dataset,
                                     batch_size=self.config.eval_batch,
                                     num_workers=self.config.eval_workers,
                                     pin_memory=True)

        #
        # Model adjustment
        #
        target_probs = class_distribution(train_targets, self.num_classes)
        init_softmax_logits(self.model.fc.bias, torch.from_numpy(target_probs))

    def configure_optimizers(self):
        optimizer = AdamW(self.optimizer_params)

        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=int(round(self.hparams['lr_t0'])),
                                                T_mult=int(round(self.hparams['lr_f'])))
        scheduler = WarmupScheduler(optimizer, epochs=self.hparams['lr_warmup'], next_scheduler=scheduler)

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

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

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
            prefix = 'train'
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
                prefix = 'train'
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
    def load_dataset(config: Dict[str, str], target_classes: List[str], store_transform: Optional[Transform] = None) -> LMDBImageDataset:
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
    def load_classes(path: str) -> List[str]:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            return [line for line in lines if line != '']

    # def score_blend(device: Union[torch.device, int],
    #                 models: List[Module],
    #                 loader: DataLoader,
    #                 eval_loss: Module,
    #                 display_progress: bool = True) -> Dict[str, Union[int, float]]:
    #     assert len(models) > 0
    #
    #     y = None
    #     y_probs_list = []
    #
    #     for model in models:
    #         model_device = model.to(device)
    #
    #         handler = PredictEpochHandler()
    #         results = run_epoch(device, model_device, loader, None, None, None, handler, display_progress=display_progress)
    #
    #         order = torch.argsort(results.indices)
    #         y_ordered = torch.index_select(results.y, dim=0, index=order)
    #         y_probs_ordered = torch.index_select(results.y_probs, dim=0, index=order)
    #
    #         if y is None:
    #             y = y_ordered
    #         else:
    #             assert bool(torch.eq(y, y_ordered).all())
    #
    #         y_probs_list.append(y_probs_ordered)
    #
    #         del model
    #
    #     y_probs_stack = torch.stack(y_probs_list, dim=0)
    #     y_probs_blend = torch.mean(y_probs_stack, dim=0)
    #
    #     y_hat = torch.argmax(y_probs_blend, dim=1)
    #     loss_value = float(eval_loss(y_probs_blend, y))
    #
    #     correct = float(torch.sum(y_hat == y))
    #     total = float(y.shape[0])
    #
    #     acc = correct / total
    #
    #     return {'loss': loss_value, 'acc': acc}

    # def score_model_blend(pipe_config: PipelineTestConfig,
    #                       dataset_config: DatasetConfig,
    #                       device: Union[torch.device, int],
    #                       paths: List[Tuple[str, str]]):
    #     #
    #     # Initialization
    #     #
    #     target_classes = load_classes(dataset_config.classes_path)
    #     test_loader = prepare_test_data(pipe_config, dataset_config, target_classes)
    #
    #     #
    #     # Snapshot loading
    #     #
    #     models = []
    #     eval_loss = None
    #     for directory, snapshot in paths:
    #         snapshot = Snapshotter.load_snapshot(directory, snapshot)
    #         model = snapshot.model.cpu()
    #
    #         models.append(model)
    #         eval_loss = snapshot.eval_loss
    #
    #         del snapshot
    #
    #     return score_blend(device, models, test_loader, eval_loss, display_progress=pipe_config.display_progress)


def run_trial(tracker: Tracker, snapshot_dir: str, tensorboard_root: str, experiment: str, device: int,
              config: AircraftClassificationConfig, max_epochs: int, hparams: Dict[str, float]) -> Dict[str, float]:
    trial_tracker = tracker.new_trial(hparams)
    snapshot_trial = os.path.join(snapshot_dir, trial_tracker.trial)

    pipeline = AircraftClassificationPipeline(config=config, hparams=hparams)
    logger = TensorBoardLogger(save_dir=tensorboard_root, name=experiment, version=trial_tracker.trial)

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


@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device', default='0', help='device number (0, 1, ...)', type=int)
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('--rnd-seed', 'rnd_seed', default=42, help='random seed', type=int)
@click.option('--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
@click.option('--freeze', 'freeze', default=-1, help='freeze first K layers', type=int)
@click.option('--mixup', 'mixup', is_flag=True, help='apply mixup augmentation', type=bool)
@click.option('--cutmix', 'cutmix', is_flag=True, help='apply cutmix augmentation', type=bool)
@click.option('--hparams', 'hparams', default='', help='hyperparameters (k1=v1,k2=v2,...)', type=str)
def cli_trial(experiment: str, device: int, repo: str, network: str,
              rnd_seed: int, max_epochs: int, freeze: int, mixup: bool, cutmix: bool,
              hparams: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    snapshot_root = config['tracking']['snapshot_root']
    tensorboard_root = config['tracking']['tensorboard_root']
    tracker_root = config['tracking']['tracker_root']

    snapshot_dir = os.path.join(snapshot_root, experiment)
    tensorboard_dir = os.path.join(tensorboard_root, experiment)

    tracker = Tracker(tracker_root, experiment)
    launch_tensorboard(tensorboard_dir)

    pipeline_config = AircraftClassificationConfig(
        random_state_seed=rnd_seed,

        classes_path=config['datasets']['classes'],
        dataset_config=config['datasets']['train'],

        repo=repo,
        network=network,

        freeze=freeze,
        mixup=mixup,
        cutmix=cutmix
    )
    hparams_dict = parse_hparams(hparams)

    results = run_trial(tracker=tracker,
                        snapshot_dir=snapshot_dir,
                        tensorboard_root=tensorboard_root,
                        experiment=experiment,
                        device=device,
                        config=pipeline_config,
                        max_epochs=max_epochs,
                        hparams=hparams_dict)

    print()
    for k, v in results.items():
        print(f'{k} = {v:.5f}')

    input("\nTrial completed, press Enter to exit (this will terminate TensorBoard)\n")


@click.option('-e', '--experiment', 'experiment', required=True, help='experiment name', type=str)
@click.option('-d', '--device', 'device', default='0', help='device number (0, 1, ...)', type=int)
@click.option('-r', '--repo', 'repo', default='pytorch/vision:v0.4.2', help='repository (e.g. pytorch/vision:v0.4.2)', type=str)
@click.option('-n', '--network', 'network', required=True, help='network (e.g. resnet50)', type=str)
@click.option('--rnd-seed', 'rnd_seed', default=42, help='random seed', type=int)
@click.option('--space', 'space', required=True, help='search space name', type=str)
@click.option('--trials', 'trials', required=True, help='number of trials to perform', type=int)
@click.option('--max-epochs', 'max_epochs', required=True, help='max number of epochs', type=int)
@click.option('--freeze', 'freeze', default=-1, help='freeze first K layers', type=int)
@click.option('--mixup', 'mixup', is_flag=True, help='apply mixup augmentation', type=bool)
@click.option('--cutmix', 'cutmix', is_flag=True, help='apply cutmix augmentation', type=bool)
def cli_search(experiment: str, device: int, repo: str, network: str,
               rnd_seed: int, space: str, trials: int, max_epochs: int, freeze: int, mixup: bool, cutmix: bool):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    snapshot_root = config['tracking']['snapshot_root']
    tensorboard_root = config['tracking']['tensorboard_root']
    tracker_root = config['tracking']['tracker_root']

    snapshot_dir = os.path.join(snapshot_root, experiment)
    tensorboard_dir = os.path.join(tensorboard_root, experiment)

    tracker = Tracker(tracker_root, experiment)
    launch_tensorboard(tensorboard_dir)

    pipeline_config = AircraftClassificationConfig(
        random_state_seed=rnd_seed,

        classes_path=config['datasets']['classes'],
        dataset_config=config['datasets']['train'],

        repo=repo,
        network=network,

        freeze=freeze,
        mixup=mixup,
        cutmix=cutmix
    )

    def run_trial_hparams(hparams: Dict[str, float]):
        results = run_trial(tracker=tracker,
                            snapshot_dir=snapshot_dir,
                            tensorboard_root=tensorboard_root,
                            experiment=experiment,
                            device=device,
                            config=pipeline_config,
                            max_epochs=max_epochs,
                            hparams=hparams)

        return results['min_val_loss']

    snapshot_root = config['tracking']['snapshot_root']
    tensorboard_root = config['tracking']['tensorboard_root']
    tracker_root = config['tracking']['tracker_root']

    snapshot_dir = os.path.join(snapshot_root, experiment)
    tensorboard_dir = os.path.join(tensorboard_root, experiment)

    tracker = Tracker(tracker_root, experiment)
    launch_tensorboard(tensorboard_dir)

    space_dict = make_space_dict()
    fmin(run_trial_hparams, space=space_dict[space], algo=hyperopt.rand.suggest, max_evals=trials)

    input("\nSearch completed, press Enter to exit (this will terminate TensorBoard)\n")
