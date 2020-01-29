from typing import Dict, Union

import cv2
import torch
from albumentations import Compose, HorizontalFlip, ToGray
from albumentations import Rotate
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from wheel5.model import fit
from wheel5.tracking import Tracker
from wheel5.transforms import SquarePaddedResize

from data import load_data


def prepare_data(datasets_config: Dict):
    lmdb_transform = SquarePaddedResize(size=224)

    aug_transform = Compose([
        HorizontalFlip(p=0.5),
        ToGray(p=0.1),
        Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0)
    ])

    model_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return load_data(datasets_config,
                     lmdb_transform=lmdb_transform,
                     aug_transform=aug_transform,
                     model_transform=model_transform)


def fit_resnet18(hparams: Dict[str, float],
                 device: Union[torch.device, int],
                 tracker: Tracker,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 classes: int,
                 max_epochs: int,
                 display_progress: bool) -> Dict[str, float]:
    # Model preparation
    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True, verbose=False)

    old_fc = model.fc
    model.fc = nn.Linear(in_features=old_fc.in_features, out_features=classes)

    model.type(torch.cuda.FloatTensor)
    model.to(device)

    loss = nn.CrossEntropyLoss()

    params_old = list([param for name, param in model.named_parameters() if not name.startswith('fc.')])
    params_new = list([param for name, param in model.named_parameters() if name.startswith('fc.')])
    optimizer_params = [
        {'params': params_old, 'lr': hparams['lrA'], 'weight_decay': hparams['wdA']},
        {'params': params_new, 'lr': hparams['lrB'], 'weight_decay': hparams['wdB']}
    ]
    optimizer = optim.AdamW(optimizer_params)

    # Training setup
    trial_tracker = tracker.new_trial(hparams)
    fit(device, model, train_loader, val_loader, loss, optimizer, num_epochs=max_epochs, tracker=trial_tracker, display_progress=display_progress)

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
