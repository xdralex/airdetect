import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from typing import List

import albumentations as albu
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from dacite import from_dict
from matplotlib.figure import Figure
from numpy.random.mtrand import RandomState
from torch import nn
from torch.nn import Parameter, CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import transforms

import wheel5.transforms.albumentations as albu_ext
import wheel5.transforms.torchvision as torchviz_ext
from wheel5.dataset import ImageOneHotDataset, ImageMixupDataset, ImageHeatmapDataset, ImageAttentiveCutMixDataset
from wheel5.dataset import TransformDataset, AlbumentationsDataset, AlbumentationsTransform
from wheel5.dataset import targets
from wheel5.loss import SoftLabelCrossEntropyLoss
from wheel5.metering import ReservoirSamplingMeter, ArrayAccumMeter
from wheel5.metrics import ExactMatchAccuracy, DiceAccuracy
from wheel5.nn import init_softmax_logits, ParamGroup
from wheel5.scheduler import WarmupScheduler
from wheel5.storage import LMDBDict
from wheel5.tasks.classification import class_distribution
from wheel5.tracking import ProbesInterface
from wheel5.tricks.gradcam import GradCAM, GradCAMpp, logit_to_score
from wheel5.tricks.heatmap import normalize_heatmap, upsample_heatmap, heatmap_to_selection_mask
from wheel5.tricks.moments import moex
from wheel5.viz import draw_confusion_matrix
from wheel5.storage import HeatmapLMDBDict
from .util import check_flag
from ..data import load_classes, ClassifierDatasetConfig, load_classifier_dataset


@dataclass
class AircraftClassifierConfig:
    random_state_seed: int

    classes_path: str
    dataset_config: Dict[str, str]
    heatmaps_path: Optional[str]

    repo: str
    network: str

    kv: Dict[str, float]

    logging_sampling_full: bool = False
    logging_samples: int = 8

    val_split: float = 0.2
    train_sample: float = 0.25

    train_batch: int = 32  # TODO: grad_batch
    train_workers: int = 4  # TODO: grad_workers
    eval_batch: int = 256
    eval_workers: int = 4


class PredictionMeter(object):
    def __init__(self):
        self.y_class = ArrayAccumMeter()
        self.y_class_hat = ArrayAccumMeter()


@dataclass
class Sample:
    x: torch.Tensor
    y: torch.Tensor


class AircraftClassifier(pl.LightningModule, ProbesInterface):

    def __init__(self, hparams: Dict):
        super(AircraftClassifier, self).__init__()

        self.journal = logging.getLogger(__name__)

        self.hparams = hparams
        self.config = from_dict(AircraftClassifierConfig, hparams)

        self.random_state = RandomState(self.config.random_state_seed)

        #
        # Statistics
        #
        self.epoch_fit_metrics = None
        self.epoch_samples: Dict[str, ReservoirSamplingMeter] = {}
        self.epoch_predictions: Dict[str, PredictionMeter] = {}

        self.training_outputs = []

        #
        # Model
        #
        self.target_classes = load_classes(self.config.classes_path)
        self.num_classes = len(self.target_classes)

        self.model = torch.hub.load(self.config.repo, self.config.network, pretrained=True, verbose=False)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.num_classes)
        self.group_names, self.optimizer_params = self.adjust_model_params()

        self.train_accuracy = DiceAccuracy()
        self.train_loss = SoftLabelCrossEntropyLoss(
            smooth_factor=self.config.kv['x_lbs'],
            smooth_dist=torch.full([self.num_classes], fill_value=1.0 / self.num_classes))

        self.eval_accuracy = ExactMatchAccuracy()
        self.eval_loss = CrossEntropyLoss()

        #
        # Transforms
        #
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        mean_color = tuple([int(round(c * 255)) for c in normalize_mean])

        self.initial_transform = AlbumentationsTransform(albu.Compose([
            albu_ext.Rescale(scale=0.5, interpolation=cv2.INTER_AREA),
            albu_ext.PadToSquare(fill=mean_color)
        ]))

        self.train_transform_mix = albu.Compose([
            albu.ShiftScaleRotate(shift_limit=0.1,
                                  scale_limit=(-0.15, 0.15),
                                  rotate_limit=20,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  value=mean_color,
                                  interpolation=cv2.INTER_AREA,
                                  p=1.0),
            albu.Resize(height=224, width=224, interpolation=cv2.INTER_AREA),
            albu.HorizontalFlip(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1.0)
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
            albu.Resize(height=224, width=224, interpolation=cv2.INTER_AREA)
        ])

        self.model_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

        self.sample_transform = torchviz_ext.InvNormalize(mean=normalize_mean, std=normalize_std)

        #
        # DataLoaders
        #
        self.train_root_dataset = None
        self.train_orig_loader = None
        self.val_loader = None

    def prepare_data(self):
        #
        # Dataset loading
        #
        fit_dataset = self.load_dataset(config=ClassifierDatasetConfig.from_dict(self.config.dataset_config), name='fit')

        #
        # Split into train/validation
        #
        fit_indices = list(range(0, len(fit_dataset)))
        self.random_state.shuffle(fit_indices)

        val_divider = int(np.round(self.config.val_split * len(fit_indices)))
        self.train_root_dataset = Subset(fit_dataset, fit_indices[val_divider:])
        val_dataset = Subset(fit_dataset, fit_indices[:val_divider])

        self.journal.info(f'Datasets: train_root[{len(self.train_root_dataset)}], val[{len(val_dataset)}]')

        #
        # Val loaders
        #
        train_root_indices = list(range(0, len(self.train_root_dataset)))
        train_sample_size = int(len(train_root_indices) * self.config.train_sample)
        train_sample_indices = self.random_state.choice(train_root_indices, size=train_sample_size, replace=False)
        train_orig_dataset = Subset(self.train_root_dataset, train_sample_indices)

        self.train_orig_loader = self.prepare_eval_loader(train_orig_dataset, name='train_orig')
        self.val_loader = self.prepare_eval_loader(val_dataset, name='val')

        #
        # Model adjustment
        #
        train_targets = targets(self.train_root_dataset)
        target_probs = class_distribution(train_targets, self.num_classes)
        init_softmax_logits(self.model.fc.bias, torch.from_numpy(target_probs))

    def load_dataset(self, config: ClassifierDatasetConfig, name: str = ''):
        return load_classifier_dataset(config=config,
                                       target_classes=self.target_classes,
                                       transform=self.initial_transform,
                                       name=name)

    def train_dataloader(self) -> DataLoader:
        train_dataset = ImageOneHotDataset(self.train_root_dataset, self.num_classes, name='train-one_hot')

        cutmix_on = check_flag(self.config.kv, 'x_cut')
        mixup_on = check_flag(self.config.kv, 'x_mxp')

        if cutmix_on and mixup_on:
            selector = 'mixup' if self.trainer.current_epoch % 2 == 0 else 'cutmix'
        elif cutmix_on:
            selector = 'cutmix'
        elif mixup_on:
            selector = 'mixup'
        else:
            selector = 'none'

        if selector == 'cutmix':
            cutmix_alpha = self.config.kv['x_cut_a']
            q_min = self.config.kv.get('x_cut_q0', 0.0)
            q_max = self.config.kv.get('x_cut_q1', 1.0)

            inter_mode = 'bilinear' if check_flag(self.config.kv, 'x_cut_i') else 'nearest'

            heatmaps = HeatmapLMDBDict(LMDBDict(self.config.heatmaps_path))
            train_dataset = ImageHeatmapDataset(train_dataset, heatmaps, inter_mode=inter_mode, name='train-heatmap')
            train_dataset = AlbumentationsDataset(train_dataset, self.train_transform_mix, use_mask=True, name='train-aug1')
            train_dataset = ImageAttentiveCutMixDataset(train_dataset, alpha=cutmix_alpha, q_min=q_min, q_max=q_max, name='train-cutmix')
        elif selector == 'mixup':
            mixup_alpha = self.config.kv['x_mxp_a']

            train_dataset = AlbumentationsDataset(train_dataset, self.train_transform_mix, use_mask=False, name='train-aug1')
            train_dataset = ImageMixupDataset(train_dataset, alpha=mixup_alpha, name='train-mixup')
        else:
            train_dataset = AlbumentationsDataset(train_dataset, self.train_transform_mix, use_mask=False, name='train-aug1')

        train_dataset = AlbumentationsDataset(train_dataset, self.train_transform_final, name='train-aug2')
        train_dataset = TransformDataset(train_dataset, self.model_transform, name='train-model')

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.config.train_batch,
                                  num_workers=self.config.train_workers,
                                  pin_memory=True,
                                  shuffle=True)

        return train_loader

    def val_dataloader(self) -> List[DataLoader]:
        return [self.val_loader, self.train_orig_loader]

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
                if index < int(round(self.config.kv['nn_frz'])):
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

    def prepare_grad_loader(self, dataset: Dataset, name: str = ''):
        dataset = AlbumentationsDataset(dataset, self.eval_transform, name=f'{name}-eval')
        dataset = TransformDataset(dataset, self.model_transform, name=f'{name}-model')

        return DataLoader(dataset,
                          batch_size=self.config.train_batch,
                          num_workers=self.config.train_workers,
                          pin_memory=True)

    def prepare_eval_loader(self, dataset: Dataset, name: str = ''):
        dataset = AlbumentationsDataset(dataset, self.eval_transform, name=f'{name}-eval')
        dataset = TransformDataset(dataset, self.model_transform, name=f'{name}-model')

        return DataLoader(dataset,
                          batch_size=self.config.eval_batch,
                          num_workers=self.config.eval_workers,
                          pin_memory=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.optimizer_params)

        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0=int(round(self.config.kv['lr_t0'])),
                                                T_mult=int(round(self.config.kv['lr_f'])))
        scheduler = WarmupScheduler(optimizer, epochs=self.config.kv['lr_w'], next_scheduler=scheduler)

        return [optimizer], [scheduler]

    def on_epoch_start(self):
        self.epoch_fit_metrics = {}

        self.epoch_samples = {}
        self.epoch_predictions = {}

        self.training_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Dict:
        x, y, *_ = batch

        if check_flag(self.config.kv, 'x_moex'):
            moex_lambda = self.config.kv['x_moex_l']
            moex_layer: nn.Module = self.model.layer1

            perm = torch.randperm(x.shape[0], device=x.device)

            def moex_hook(_: nn.Module, input: torch.Tensor) -> Optional[Tuple[torch.Tensor]]:
                input, = input
                return moex(input, perm),

            moex_handle = moex_layer.register_forward_pre_hook(moex_hook)

            z = self.forward(x)
            y = y * (1 - moex_lambda) + y[perm] * moex_lambda

            moex_handle.remove()
        else:
            z = self.forward(x)

        y_probs_hat = torch.exp(log_softmax(z, dim=1))

        loss = self.train_loss(z, y)
        prefix = 'train'
        numer, denom = self.train_accuracy(y_probs_hat, y, prefix)

        output = {
            f'{prefix}_loss': loss,
            f'{prefix}_numer': numer,
            f'{prefix}_denom': denom
        }
        self.training_outputs.append(output)

        self.add_epoch_samples('train', batch_idx, x, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int) -> Dict:
        x, y, *_ = batch

        z = self.forward(x)
        y_probs_hat = torch.exp(log_softmax(z, dim=1))
        y_class_hat = torch.argmax(y_probs_hat, dim=1)

        if dataloader_idx == 0:  # val_loader
            prefix = 'val'
        elif dataloader_idx == 1:  # train_orig_loader
            prefix = 'train-orig'
        else:
            raise AssertionError(f'Invalid dataloader index: {dataloader_idx}')

        loss = self.eval_loss(z, y)
        numer, denom = self.eval_accuracy(y_class_hat, y, prefix)

        self.add_epoch_samples(prefix, batch_idx, x, y)
        self.add_epoch_predictions(prefix, y, y_class_hat)

        return {
            f'{prefix}_loss': loss,
            f'{prefix}_numer': numer,
            f'{prefix}_denom': denom
        }

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        metrics = OrderedDict()

        outputs_by_prefix = OrderedDict()
        outputs_by_prefix['train'] = self.training_outputs

        for dataloader_idx, output in enumerate(outputs):
            if dataloader_idx == 0:  # val_loader
                prefix = 'val'
            elif dataloader_idx == 1:  # train_orig_loader
                prefix = 'train-orig'
            else:
                raise AssertionError(f'Invalid dataloader index: {dataloader_idx}')

            outputs_by_prefix[prefix] = output

        for prefix, output in outputs_by_prefix.items():
            loss = torch.stack([x[f'{prefix}_loss'] for x in output]).mean()

            numer_sum = torch.stack([x[f'{prefix}_numer'] for x in output]).sum()
            denom_sum = torch.stack([x[f'{prefix}_denom'] for x in output]).sum()
            accuracy = numer_sum / float(denom_sum)

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

    def add_epoch_samples(self, name: str, batch_idx: int, x: torch.Tensor, y: torch.Tensor):
        if batch_idx == 0 or self.config.logging_sampling_full:
            if name not in self.epoch_samples:
                self.epoch_samples[name] = ReservoirSamplingMeter(k=self.config.logging_samples)

            elements = []
            for i in range(0, y.shape[0]):
                sample = Sample(x=x[i], y=y[i])
                elements.append(sample)

            meter = self.epoch_samples[name]
            meter.add(elements)

    def add_epoch_predictions(self, name: str, y_class: torch.Tensor, y_class_hat: torch.Tensor):
        if name not in self.epoch_predictions:
            self.epoch_predictions[name] = PredictionMeter()

        meter = self.epoch_predictions[name]
        meter.y_class.add(y_class)
        meter.y_class_hat.add(y_class_hat)

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
        figures = {}

        for name, meter in self.epoch_predictions.items():
            y_class = meter.y_class.value().cpu().numpy()
            y_class_hat = meter.y_class_hat.value().cpu().numpy()

            # TODO: draw mispredicted images
            cm_fig = draw_confusion_matrix(self.target_classes, y_true=y_class, y_pred=y_class_hat)
            figures[f'confusion_matrix/{name}'] = cm_fig

            #
            #         mismatch_figs = visualize_top_errors(classes,
            #                                              y_true=prediction['y'].numpy(),
            #                                              y_pred=prediction['y_hat'].numpy(),
            #                                              image_indices=prediction['indices'].numpy(),
            #                                              image_dataset=TransformDataset(prediction_dataset, viz_transform))

        return figures

    def introspect_cam(self,
                       batch,
                       class_selector: str,
                       layer: Optional[nn.Module] = None,
                       cam_generator: str = 'gradcampp',
                       inter_mode: Optional[str] = None,
                       cutoff_ratio: Optional[float] = None) -> torch.Tensor:
        if layer is None:
            layer = self.model.layer4

        x, y, *_ = batch
        _, _, h, w = x.shape

        if class_selector == 'pred':
            score_fn = logit_to_score()
        elif class_selector == 'true':
            score_fn = logit_to_score(y)
        else:
            raise AssertionError(f'Invalid class selector: {class_selector}')

        if cam_generator == 'gradcam':
            with GradCAM(self.model, layer, score_fn) as grad_cam:
                heatmap = grad_cam.generate(x)
        elif cam_generator == 'gradcampp':
            with GradCAMpp(self.model, layer, score_fn) as grad_cam:
                heatmap = grad_cam.generate(x)
        else:
            raise AssertionError(f'Invalid CAM mode: {cam_generator}')

        heatmap = normalize_heatmap(heatmap)

        if inter_mode is not None:
            heatmap = upsample_heatmap(heatmap, h, w, inter_mode)

        if cutoff_ratio is not None:
            heatmap = heatmap_to_selection_mask(heatmap, cutoff_ratio).int()

        return heatmap

    def get_tqdm_dict(self):
        d = dict(super(AircraftClassifier, self).get_tqdm_dict())
        del d['v_num']
        return d
