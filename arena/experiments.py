from typing import Dict, Union, NamedTuple

import albumentations as albu
import cv2
import numpy as np
import torch
import wheel5.transforms as wheeltr
from PIL import Image
from numpy.random.mtrand import RandomState
from torch import nn
from torch import optim
from torchvision import transforms
from wheel5.dataset import TransformDataset, AlbumentationsDataset
from wheel5.model import fit
from wheel5.tracking import Tracker

from data import DataBundle, load_dataset, prepare_train_bundle, prepare_eval_bundle, prepare_test_bundle, Transform


def make_target_classes():
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


class TransformsBundle(NamedTuple):
    store: Transform
    aug: albu.BasicTransform
    model: Transform
    sample: Transform


def make_transforms_bundle():
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    # Transform applied when the data is preprocessed into the store
    store_transform = transforms.Compose([
        wheeltr.Rescale(scale=0.5, interpolation=Image.LANCZOS),
        wheeltr.PadToSquare()
    ])

    # Transform augmenting the data
    aug_transform = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.OneOf([
            albu.ToGray(p=0.1),
            albu.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.9)
        ]),
        albu.CoarseDropout(max_holes=10, max_height=10, max_width=10, min_holes=5, min_height=5, min_width=5, fill_value=0, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.1,
                              scale_limit=(-0.2, 0.1),
                              rotate_limit=15,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=0,
                              interpolation=cv2.INTER_LANCZOS4,
                              p=1.0)
    ])

    # Transform preparing the data to be processed by the model
    model_transform = transforms.Compose([
        wheeltr.SquarePaddedResize(size=224, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    # Transform unnormalizing the data to be displayed by TensorBoard
    sample_transform = wheeltr.InvNormalize(mean=normalize_mean, std=normalize_std)

    return TransformsBundle(
        store=store_transform,
        aug=aug_transform,
        model=model_transform,
        sample=sample_transform)


TARGET_CLASSES = make_target_classes()
TRANSFORMS_BUNDLE = make_transforms_bundle()
CONTROL_SIZE = 64
VAL_SPLIT = 0.2


def prepare_model_test_bundle(dataset_config: Dict[str, str]) -> DataBundle:
    dataset = load_dataset(dataset_config, TARGET_CLASSES, TRANSFORMS_BUNDLE.store)
    bundle = prepare_test_bundle(TransformDataset(dataset, TRANSFORMS_BUNDLE.model))
    return bundle


def prepare_model_fit_bundles(dataset_config: Dict[str, str]):
    dataset = load_dataset(dataset_config, TARGET_CLASSES, TRANSFORMS_BUNDLE.store)

    # Indices
    random_state = np.random.RandomState(715)

    indices = list(range(0, len(dataset)))
    random_state.shuffle(indices)

    ctrl_indices, train_val_indices = indices[:CONTROL_SIZE], indices[CONTROL_SIZE:]

    divider = int(np.round(VAL_SPLIT * len(train_val_indices)))
    train_indices, val_indices = indices[divider:], indices[:divider]

    # Bundles
    train_dataset = AlbumentationsDataset(dataset, TRANSFORMS_BUNDLE.aug)
    train_dataset = TransformDataset(train_dataset, TRANSFORMS_BUNDLE.model)
    train_bundle = prepare_train_bundle(train_dataset, list(train_indices))

    val_dataset = TransformDataset(dataset, TRANSFORMS_BUNDLE.model)
    val_bundle = prepare_eval_bundle(val_dataset, list(val_indices), randomize=True)

    ctrl_dataset = TransformDataset(dataset, TRANSFORMS_BUNDLE.model)
    ctrl_bundle = prepare_eval_bundle(ctrl_dataset, list(ctrl_indices), randomize=False)

    return train_bundle, val_bundle, ctrl_bundle


def fit_resnet(dataset_config: Dict[str, str],
               nn_name: str,
               hparams: Dict[str, float],
               device: Union[torch.device, int],
               tracker: Tracker,
               max_epochs: int,
               display_progress: bool) -> Dict[str, float]:
    # Model preparation
    model = torch.hub.load('pytorch/vision:v0.4.2', nn_name, pretrained=True, verbose=False)

    old_fc = model.fc
    model.fc = nn.Linear(in_features=old_fc.in_features, out_features=len(TARGET_CLASSES))

    model.type(torch.cuda.FloatTensor)
    model.to(device)

    loss = nn.CrossEntropyLoss()

    freeze = int(round(hparams['freeze']))
    for index, (name, child) in enumerate(model.named_children()):
        if index < freeze:
            for param in child.parameters(recurse=True):
                param.requires_grad = False

    grad_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    params_old = list([param for name, param in grad_params if not name.startswith('fc.')])
    params_new = list([param for name, param in grad_params if name.startswith('fc.')])
    optimizer_params = [
        {'params': params_old, 'lr': hparams['lrA'], 'weight_decay': hparams['wdA']},
        {'params': params_new, 'lr': hparams['lrB'], 'weight_decay': hparams['wdB']}
    ]
    optimizer = optim.AdamW(optimizer_params)

    # Data loading
    train_bundle, val_bundle, ctrl_bundle = prepare_model_fit_bundles(dataset_config)

    # Training setup
    trial_tracker = tracker.new_trial(hparams)
    fit(device,
        model,
        train_bundle.loader,
        val_bundle.loader,
        ctrl_bundle.loader,
        loss,
        optimizer,
        num_epochs=max_epochs,
        tracker=trial_tracker,
        display_progress=display_progress,
        sampled_epochs=10)

    # Reporting
    metrics_df = trial_tracker.metrics_df
    results = {
        'hp/best_val_acc': metrics_df['val_acc'].max(),
        'hp/best_val_loss': metrics_df['val_loss'].min(),
        'hp/final_val_acc': metrics_df['val_acc'].iloc[-1],
        'hp/final_val_loss': metrics_df['val_loss'].iloc[-1],

        'hp/best_train_acc': metrics_df['train_acc'].max(),
        'hp/best_train_loss': metrics_df['train_loss'].min(),
        'hp/final_train_acc': metrics_df['train_acc'].iloc[-1],
        'hp/final_train_loss': metrics_df['train_loss'].iloc[-1],
    }
    trial_tracker.trial_completed(results)

    return results
