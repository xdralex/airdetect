import logging
import os
import sys
from dataclasses import asdict
from typing import Dict, Optional, Union
from typing import List, Callable

import torch
from PIL.Image import Image as Img
from dacite import from_dict
from matplotlib.figure import Figure
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import rank_zero_only
from torch.nn.functional import log_softmax
from tqdm import tqdm

from wheel5.storage import LMDBDict
from wheel5.tracking import Tracker, TensorboardLogging, StatisticsTracking, CheckpointPattern
from wheel5.util import shape
from wheel5.viz import draw_heatmap, HeatmapEntry, HeatmapModeColormap, HeatmapModeBloom
from wheel5.viz.predictions import draw_classes
from .classifier import AircraftClassifierConfig, AircraftClassifier
from ..data import ClassifierDatasetConfig
from ..storage import HeatmapLMDBDict

Transform = Callable[[Img], Img]


class TensorboardHparamsLogger(TensorBoardLogger):
    def __init__(self, save_dir: str, name: Optional[str] = "default", version: Optional[Union[int, str]] = None, **kwargs):
        super(TensorboardHparamsLogger, self).__init__(save_dir, name, version, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params: Dict) -> None:
        config = from_dict(AircraftClassifierConfig, params)
        super(TensorboardHparamsLogger, self).log_hyperparams(config.kv)


def visualize_predictions(dataset_config: Dict[str, str],
                          snapshot_path: str,
                          device: int,
                          samples: int) -> Figure:
    device = torch.device(f'cuda:{device}')

    model = AircraftClassifier.load_from_checkpoint(snapshot_path, map_location=device)
    model.to(device)
    model.freeze()
    model.eval()

    dataset = model.load_dataset(config=ClassifierDatasetConfig.from_dict(dataset_config))
    loader = model.prepare_eval_loader(dataset)

    x_list = []
    z_list = []
    count = 0

    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device)
            z = model.forward(x)

            x_list.append(x)
            z_list.append(z)

            count += x.shape[0]
            if count >= samples:
                break

    x = torch.cat(x_list)[0:samples]
    z = torch.cat(z_list)[0:samples]

    y_probs_hat = torch.exp(log_softmax(z, dim=1))
    x_sample = torch.stack([model.sample_transform(x[i]) for i in range(0, samples)])

    return draw_classes(x_sample, y_probs_hat, model.target_classes)


def visualize_heatmap(dataset_config: Dict[str, str],
                      snapshot_path: str,
                      device: int,
                      samples: int,
                      no_actual: bool = False,
                      inter_mode: str = 'nearest',
                      cutoff_ratio: Optional[float] = None) -> Figure:
    device = torch.device(f'cuda:{device}')

    model = AircraftClassifier.load_from_checkpoint(snapshot_path, map_location=device)
    model.to(device)
    model.unfreeze()
    model.eval()

    dataset = model.load_dataset(config=ClassifierDatasetConfig.from_dict(dataset_config))
    loader = model.prepare_grad_loader(dataset)

    x_list = []
    y_list = []
    z_list = []
    count = 0

    for x, y, *_ in loader:
        x = x.to(device)
        y = y.to(device)
        z = model.forward(x)

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

        count += x.shape[0]
        if count >= samples:
            break

    x = torch.cat(x_list)[0:samples]
    y = torch.cat(y_list)[0:samples]
    z = torch.cat(z_list)[0:samples]

    y_probs_hat = torch.exp(log_softmax(z, dim=1))
    y_class_hat = torch.argmax(y_probs_hat, dim=1)

    batch = (x, y, None)

    x_sample = torch.stack([model.sample_transform(x[i]) for i in range(0, samples)])

    if no_actual:
        mask_hat = model.introspect_cam(batch, class_selector='pred', inter_mode=inter_mode, cutoff_ratio=cutoff_ratio)
        entries = [
            HeatmapEntry('predicted', y_class_hat, mask_hat, mode=HeatmapModeColormap()),
            HeatmapEntry('predicted', y_class_hat, mask_hat, mode=HeatmapModeBloom())
        ]
    else:
        mask = model.introspect_cam(batch, class_selector='true', inter_mode=inter_mode, cutoff_ratio=cutoff_ratio)
        mask_hat = model.introspect_cam(batch, class_selector='pred', inter_mode=inter_mode, cutoff_ratio=cutoff_ratio)
        entries = [
            HeatmapEntry('actual', y, mask, mode=HeatmapModeColormap()),
            HeatmapEntry('predicted', y_class_hat, mask_hat, mode=HeatmapModeColormap()),
            HeatmapEntry('actual', y, mask, mode=HeatmapModeBloom()),
            HeatmapEntry('predicted', y_class_hat, mask_hat, mode=HeatmapModeBloom())
        ]

    return draw_heatmap(x_sample, entries, model.target_classes)


def build_heatmaps(dataset_config: Dict[str, str],
                   snapshot_path: str,
                   heatmap_path: str,
                   device: int,
                   show_progress: bool = True):
    device = torch.device(f'cuda:{device}')

    model = AircraftClassifier.load_from_checkpoint(snapshot_path, map_location=device)
    model.to(device)
    model.unfreeze()
    model.eval()

    dataset = model.load_dataset(config=ClassifierDatasetConfig.from_dict(dataset_config))
    loader = model.prepare_grad_loader(dataset)

    logger = logging.getLogger(f'{__name__}')

    with LMDBDict(heatmap_path) as lmdb_dict:
        heatmap_db = HeatmapLMDBDict(lmdb_dict)

        with tqdm(total=len(loader), disable=not show_progress, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f'Building heatmaps')

            for x, y, indices, paths, *_ in loader:
                x = x.to(device)
                y = y.to(device)

                batch = (x, y, indices)
                mask = model.introspect_cam(batch, class_selector='true')

                assert indices.shape[0] == mask.shape[0]
                mask = mask.cpu().numpy()

                for i in range(0, indices.shape[0]):
                    heatmap_db[paths[i]] = mask[i]
                    logger.info(f'heatmap: #{paths[i]} - {shape(mask[i])}')

                progress_bar.update()


def eval_blend(dataset_config: Dict[str, str],
               device: int,
               snapshot_paths: List[str],
               show_progress: bool = True) -> Dict[str, float]:
    device = torch.device(f'cuda:{device}')

    torch.set_printoptions(linewidth=250, edgeitems=10)

    with torch.no_grad():
        loader = None

        y_only = None
        y_probs_hat_list = []
        for i, snapshot_path in enumerate(snapshot_paths):
            model = AircraftClassifier.load_from_checkpoint(snapshot_path, map_location=device)
            model.to(device)
            model.freeze()
            model.eval()

            if loader is None:
                dataset = model.load_dataset(config=ClassifierDatasetConfig.from_dict(dataset_config))
                loader = model.prepare_eval_loader(dataset)

            z_list = []
            y_list = []
            with tqdm(total=len(loader), disable=not show_progress, file=sys.stdout) as progress_bar:
                progress_bar.set_description(f'Evaluating model {i + 1}')

                for x, y, *_ in loader:
                    x = x.to(device)
                    y = y.to(device)

                    z_list.append(model.forward(x))
                    y_list.append(y)

                    progress_bar.update()

            z = torch.cat(z_list)
            y = torch.cat(y_list)
            y_probs_hat = torch.exp(log_softmax(z, dim=1))

            y_probs_hat_list.append(y_probs_hat)
            if y_only is None:
                y_only = y
            del model

        y_probs_hat_stack = torch.stack(y_probs_hat_list, dim=0)
        y_probs_hat_blend = torch.mean(y_probs_hat_stack, dim=0)
        y_class_hat = torch.argmax(y_probs_hat_blend, dim=1)

        model = AircraftClassifier.load_from_checkpoint(snapshot_paths[0], map_location=device)
        model.to(device)
        model.freeze()
        model.eval()

        numer, denom = model.eval_accuracy(y_class_hat, y_only, 'test')
        accuracy = float(numer) / float(denom)

        return {
            'acc': accuracy
        }


def fit_trial(tracker: Tracker,
              snapshot_dir: str,
              tensorboard_root: str,
              experiment: str,
              device: int,
              config: AircraftClassifierConfig,
              max_epochs: int) -> Dict[str, float]:

    torch.set_printoptions(linewidth=250, edgeitems=10)

    trial_tracker = tracker.new_trial(config.kv)
    snapshot_trial = os.path.join(snapshot_dir, trial_tracker.trial)

    pipeline = AircraftClassifier(hparams=asdict(config))
    logger = TensorboardHparamsLogger(save_dir=tensorboard_root, name=experiment, version=trial_tracker.trial)

    checkpoint_callback = ModelCheckpoint(filepath=CheckpointPattern.pattern(snapshot_trial), monitor='val_acc', mode='max', save_top_k=1)
    early_stop_callback = False
    tensorboard_callback = TensorboardLogging()
    tracking_callback = StatisticsTracking(trial_tracker)

    trainer = Trainer(logger=logger,
                      reload_dataloaders_every_epoch=True,
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
        'min_train-orig_loss': metrics_df['train-orig_loss'].min(),
        'max_train-orig_acc': metrics_df['train-orig_acc'].max()
    }
