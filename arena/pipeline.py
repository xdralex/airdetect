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
from torchvision import transforms
from wheel5.dataset import TransformDataset, AlbumentationsDataset, ImageDataset, LMDBImageDataset, ImageOneHotDataset, ImageCutMixDataset, ImageMixupDataset, \
    SequentialSubsetSampler
from wheel5.loss import SoftLabelCrossEntropyLoss
from wheel5.metrics import ExactMatchAccuracy, JaccardAccuracy
from wheel5.model import fit
from wheel5.model import score_blend
from wheel5.nn import init_softmax_logits, ParamGroup
from wheel5.scheduler import WarmupScheduler
from wheel5.tracking import Tracker, Snapshotter

from dataset.functional import class_distribution
from .data import load_dataset, DatasetConfig, load_classes


# TODO: INTER_AREA
# TODO: use albumentations throughout?


@dataclass
class PipelineFitConfig:
    hparams: Dict[str, float]

    repo: str
    network: str

    max_epochs: int
    freeze: int
    mixup: bool
    cutmix: bool

    display_progress: bool = True
    print_model_transforms: bool = True

    ctrl_size: int = 64
    val_split: float = 0.2

    train_batch: int = 32
    train_workers: int = 4
    eval_batch: int = 256
    eval_workers: int = 4


@dataclass
class PipelineTestConfig:
    display_progress: bool = True

    eval_batch: int = 256
    eval_workers: int = 4


def make_sample_transform():
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    return torchviz_ext.InvNormalize(mean=normalize_mean, std=normalize_std)


def prepare_fit_data(pipe_config: PipelineFitConfig,
                     dataset_config: DatasetConfig,
                     target_classes: List[str],
                     random_state: RandomState) -> Tuple[LMDBImageDataset, DataLoader, DataLoader, DataLoader]:
    #
    # Transforms
    #
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    mean_color = tuple([int(round(c * 255)) for c in normalize_mean])

    store_transform = transforms.Compose([
        torchviz_ext.Rescale(scale=0.5, interpolation=Image.LANCZOS)
    ])

    train_transform = albu.Compose([
        albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1.0),
        albu.HorizontalFlip(p=0.5),

        albu_ext.PadToSquare(fill=mean_color),
        albu.ShiftScaleRotate(shift_limit=0.1,
                              scale_limit=(-0.25, 0.15),
                              rotate_limit=20,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=mean_color,
                              interpolation=cv2.INTER_AREA,
                              p=1.0),

        albu_ext.Resize(height=224, width=224, interpolation=cv2.INTER_AREA),
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
    dataset: LMDBImageDataset = load_dataset(dataset_config, target_classes, store_transform)

    #
    # Split into train/val/ctrl
    #
    indices = list(range(0, len(dataset)))
    random_state.shuffle(indices)

    ctrl_indices, train_val_indices = indices[:pipe_config.ctrl_size], indices[pipe_config.ctrl_size:]

    val_divider = int(np.round(pipe_config.val_split * len(train_val_indices)))
    train_indices, val_indices = indices[val_divider:], indices[:val_divider]

    train_dataset = cast(ImageDataset[int], Subset(dataset, train_indices))
    val_dataset = cast(ImageDataset[int], Subset(dataset, val_indices))
    ctrl_dataset = cast(ImageDataset[int], Subset(dataset, ctrl_indices))

    #
    # Train transformations
    #
    train_dataset = ImageOneHotDataset(train_dataset, len(target_classes))

    if pipe_config.cutmix:
        cutmix_alpha = pipe_config.hparams['cutmix_alpha']
        train_dataset = ImageCutMixDataset(train_dataset, alpha=cutmix_alpha, random_state=random_state)

    if pipe_config.mixup:
        mixup_alpha = pipe_config.hparams['mixup_alpha']
        train_dataset = ImageMixupDataset(train_dataset, alpha=mixup_alpha, random_state=random_state)

    train_dataset = AlbumentationsDataset(train_dataset, train_transform)
    train_dataset = TransformDataset(train_dataset, model_transform)

    #
    # Eval transformations
    #
    val_dataset = AlbumentationsDataset(val_dataset, eval_transform)
    val_dataset = TransformDataset(val_dataset, model_transform)

    ctrl_dataset = AlbumentationsDataset(ctrl_dataset, eval_transform)
    ctrl_dataset = TransformDataset(ctrl_dataset, model_transform)

    #
    # Data loading
    #
    train_sampler = SubsetRandomSampler(list(range(0, len(train_dataset))))
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              batch_size=pipe_config.train_batch,
                              num_workers=pipe_config.train_workers,
                              pin_memory=True)

    val_sampler = SubsetRandomSampler(list(range(0, len(val_dataset))))
    val_loader = DataLoader(val_dataset,
                            sampler=val_sampler,
                            batch_size=pipe_config.eval_batch,
                            num_workers=pipe_config.eval_workers,
                            pin_memory=True)

    ctrl_sampler = SequentialSubsetSampler(list(range(0, len(ctrl_dataset))))
    ctrl_loader = DataLoader(ctrl_dataset,
                             sampler=ctrl_sampler,
                             batch_size=pipe_config.eval_batch,
                             num_workers=pipe_config.eval_workers,
                             pin_memory=True)

    return dataset, train_loader, val_loader, ctrl_loader


def prepare_test_data(pipe_config: PipelineTestConfig,
                      dataset_config: DatasetConfig,
                      target_classes: List[str]) -> DataLoader:
    #
    # Transforms
    #
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    mean_color = tuple([int(round(c * 255)) for c in normalize_mean])

    store_transform = transforms.Compose([
        torchviz_ext.Rescale(scale=0.5, interpolation=Image.LANCZOS)
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
    dataset: LMDBImageDataset = load_dataset(dataset_config, target_classes, store_transform)

    #
    # Eval transformations
    #
    test_dataset = dataset
    test_dataset = AlbumentationsDataset(test_dataset, eval_transform)
    test_dataset = TransformDataset(test_dataset, model_transform)

    #
    # Data loading
    #
    test_sampler = SubsetRandomSampler(list(range(0, len(test_dataset))))
    test_loader = DataLoader(test_dataset,
                             sampler=test_sampler,
                             batch_size=pipe_config.eval_batch,
                             num_workers=pipe_config.eval_workers,
                             pin_memory=True)

    return test_loader


def adjust_model_params(pipe_config: PipelineFitConfig, model: nn.Module):
    param_groups = {
        'A': ParamGroup({'lr': pipe_config.hparams['lrA'], 'weight_decay': pipe_config.hparams['wdA']}),
        'A_no_decay': ParamGroup({'lr': pipe_config.hparams['lrA'], 'weight_decay': 0}),
        'B': ParamGroup({'lr': pipe_config.hparams['lrB'], 'weight_decay': pipe_config.hparams['wdB']}),
        'B_no_decay': ParamGroup({'lr': pipe_config.hparams['lrB'], 'weight_decay': 0})
    }

    def add_param(group_name: str, module_name: str, param_name: str, param: Parameter):
        if param.requires_grad:
            group = param_groups[group_name]
            group.params.append((f'{module_name}.{param_name}', param))

    def freeze_params():
        for index, (name, child) in enumerate(model.named_children()):
            if index < pipe_config.freeze:
                if pipe_config.print_model_transforms:
                    print(f'Freezing layer {name}')

                for param in child.parameters(recurse=True):
                    param.requires_grad = False

    def init_param_groups():
        for module_name, module in model.named_modules():
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
        if pipe_config.print_model_transforms:
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


def fit_model(pipe_config: PipelineFitConfig,
              dataset_config: DatasetConfig,
              device: Union[torch.device, int],
              tracker: Tracker) -> Dict[str, float]:
    #
    # Initialization
    #
    random_state = RandomState(42)

    target_classes = load_classes(dataset_config.classes_path)
    num_classes = len(target_classes)

    #
    # Data loading
    #
    dataset, train_loader, val_loader, ctrl_loader = prepare_fit_data(pipe_config, dataset_config, target_classes, random_state)

    #
    # Adjusting the model
    #
    model = torch.hub.load(pipe_config.repo, pipe_config.network, pretrained=True, verbose=False)

    old_fc = model.fc
    model.fc = nn.Linear(in_features=old_fc.in_features, out_features=num_classes)

    target_probs = class_distribution(dataset.targets(), num_classes)
    init_softmax_logits(model.fc.bias, torch.from_numpy(target_probs))

    model.type(torch.FloatTensor)
    model.to(device)

    group_names, optimizer_params = adjust_model_params(pipe_config, model)

    #
    # Setting up training
    #
    optimizer = AdamW(optimizer_params)
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=int(round(pipe_config.hparams['cos_t0'])),
                                            T_mult=int(round(pipe_config.hparams['cos_f'])))
    scheduler = WarmupScheduler(optimizer, epochs=3, next_scheduler=scheduler)

    smooth_dist = torch.full([num_classes], fill_value=1.0 / num_classes)
    train_loss = SoftLabelCrossEntropyLoss(smooth_factor=pipe_config.hparams['smooth'], smooth_dist=smooth_dist)
    train_accuracy = JaccardAccuracy()

    eval_loss = CrossEntropyLoss()
    eval_accuracy = ExactMatchAccuracy()

    #
    # Fitting the model
    #
    trial_tracker = tracker.new_trial(pipe_config.hparams)
    fit(device,
        model,
        classes=target_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        ctrl_loader=ctrl_loader,
        train_loss=train_loss,
        eval_loss=eval_loss,
        train_accuracy=train_accuracy,
        eval_accuracy=eval_accuracy,
        optimizer=optimizer,
        scheduler=scheduler,
        group_names=group_names,
        num_epochs=pipe_config.max_epochs,
        tracker=trial_tracker,
        display_progress=pipe_config.display_progress,
        sampled_epochs=pipe_config.max_epochs)

    metrics_df = trial_tracker.metrics_df
    results = {
        'hp/best_val_acc': metrics_df['val_acc'].max(),
        'hp/best_val_loss': metrics_df['val_loss'].min(),
        'hp/best_train_acc': metrics_df['train_acc'].max(),
        'hp/best_train_loss': metrics_df['train_loss'].min(),
    }
    trial_tracker.trial_completed(results)

    return results


def score_model_blend(pipe_config: PipelineTestConfig,
                      dataset_config: DatasetConfig,
                      device: Union[torch.device, int],
                      paths: List[Tuple[str, str]]):
    #
    # Initialization
    #
    target_classes = load_classes(dataset_config.classes_path)
    test_loader = prepare_test_data(pipe_config, dataset_config, target_classes)

    #
    # Snapshot loading
    #
    models = []
    eval_loss = None
    for directory, snapshot in paths:
        snapshot = Snapshotter.load_snapshot(directory, snapshot)
        model = snapshot.model.cpu()

        models.append(model)
        eval_loss = snapshot.eval_loss

        del snapshot

    return score_blend(device, models, test_loader, eval_loss, display_progress=pipe_config.display_progress)
