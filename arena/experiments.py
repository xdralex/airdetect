import math
from typing import Dict, Union, NamedTuple, Tuple, List, Optional

import albumentations as albu
import cv2
import numpy as np
import torch
import wheel5.transforms_albumentations as wheeltr_albu
import wheel5.transforms_torchvision as wheeltr_torch
from PIL import Image
from numpy.random.mtrand import RandomState
from torch import nn
from torch import optim
from torch.nn import Parameter, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from wheel5.dataretriever import DirectDataRetriever, MixupDataRetriever
from wheel5.dataset import TransformDataset, AlbumentationsDataset
from wheel5.loss import SmoothedCrossEntropyLoss
from wheel5.model import fit
from wheel5.model import score_blend
from wheel5.nn import init_softmax_logits
from wheel5.scheduler import WarmupScheduler
from wheel5.tracking import Tracker, Snapshotter
from wheel5.formats import TargetFormat
from wheel5.metrics import ExactMatchAccuracy, JaccardAccuracy

from data import DataBundle, load_dataset, prepare_train_bundle, prepare_eval_bundle, prepare_test_bundle, Transform



class TransformsBundle(NamedTuple):
    store: Transform
    train: albu.BasicTransform
    eval: albu.BasicTransform
    model: Transform
    sample: Optional[Transform]


class ExperimentConfig(NamedTuple):
    repo: str
    network: str
    hparams: Dict[str, float]
    max_epochs: int
    freeze: int
    mixup: bool


class ModelFitBundle(NamedTuple):
    train1: DataBundle
    train2: Optional[DataBundle]
    val: DataBundle
    ctrl: DataBundle


class ParamGroup(object):
    def __init__(self, config: Dict[str, float]):
        self.config: Dict[str, float] = config
        self.params: List[Tuple[str, Parameter]] = []

    def parameters(self) -> List[Parameter]:
        return [param for _, param in self.params]

    def __repr__(self):
        dump = 'ParamGroup(\n'

        dump += '  config:\n'
        for k, v in self.config.items():
            dump += f'    {k} -> {v:.8f}\n'

        dump += '  params:\n'
        for param_name, param in self.params:
            param_shape = 'x'.join([str(dim) for dim in param.shape])
            dump += f'    {param_name} - {param_shape}\n'

        dump += ')\n'

        return dump


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


def make_transforms_bundle():
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]

    mean_color = tuple([int(round(c * 255)) for c in normalize_mean])

    # TODO: INTER_AREA

    # Transform applied when the data is preprocessed into the store
    store_transform = transforms.Compose([
        wheeltr_torch.Rescale(scale=0.5, interpolation=Image.LANCZOS)
    ])

    # Train/Eval transforms
    train_transform = albu.Compose([
        albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1.0),
        albu.HorizontalFlip(p=0.5),

        wheeltr_albu.PadToSquare(fill=mean_color),
        albu.ShiftScaleRotate(shift_limit=0.1,
                              scale_limit=(-0.25, 0.15),
                              rotate_limit=20,
                              border_mode=cv2.BORDER_CONSTANT,
                              value=mean_color,
                              interpolation=cv2.INTER_AREA,
                              p=1.0),

        wheeltr_albu.Resize(height=224, width=224, interpolation=cv2.INTER_AREA),
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
        wheeltr_albu.PadToSquare(fill=mean_color),
        wheeltr_albu.Resize(height=224, width=224, interpolation=cv2.INTER_AREA)
    ])

    # Transform preparing the data to be processed by the model
    model_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    # Transform unnormalizing the data to be displayed by TensorBoard
    sample_transform = wheeltr_torch.InvNormalize(mean=normalize_mean, std=normalize_std)

    return TransformsBundle(
        store=store_transform,
        train=train_transform,
        eval=eval_transform,
        model=model_transform,
        sample=sample_transform)


TARGET_CLASSES = make_target_classes()
TRANSFORMS_BUNDLE = make_transforms_bundle()
CONTROL_SIZE = 64
VAL_SPLIT = 0.2


def prepare_model_test_bundle(dataset_config: Dict[str, str]) -> DataBundle:
    dataset = load_dataset(dataset_config, TARGET_CLASSES, TRANSFORMS_BUNDLE.store)
    dataset = AlbumentationsDataset(dataset, TRANSFORMS_BUNDLE.eval)
    dataset = TransformDataset(dataset, TRANSFORMS_BUNDLE.model)
    bundle = prepare_test_bundle(dataset)
    return bundle


def prepare_model_fit_bundle(dataset_config: Dict[str, str], multi_train: bool) -> ModelFitBundle:
    dataset = load_dataset(dataset_config, TARGET_CLASSES, TRANSFORMS_BUNDLE.store)

    # Indices
    random_state = np.random.RandomState(715)

    indices = list(range(0, len(dataset)))
    random_state.shuffle(indices)

    ctrl_indices, train_val_indices = indices[:CONTROL_SIZE], indices[CONTROL_SIZE:]

    divider = int(np.round(VAL_SPLIT * len(train_val_indices)))
    train_indices, val_indices = indices[divider:], indices[:divider]

    # Bundles
    train_dataset1 = AlbumentationsDataset(dataset, TRANSFORMS_BUNDLE.train)
    train_dataset1 = TransformDataset(train_dataset1, TRANSFORMS_BUNDLE.model)
    train_bundle1 = prepare_train_bundle(train_dataset1, list(train_indices))

    if multi_train:
        train_dataset2 = AlbumentationsDataset(dataset, TRANSFORMS_BUNDLE.train)
        train_dataset2 = TransformDataset(train_dataset2, TRANSFORMS_BUNDLE.model)
        train_bundle2 = prepare_train_bundle(train_dataset2, list(train_indices))
    else:
        train_bundle2 = None

    val_dataset = AlbumentationsDataset(dataset, TRANSFORMS_BUNDLE.eval)
    val_dataset = TransformDataset(val_dataset, TRANSFORMS_BUNDLE.model)
    val_bundle = prepare_eval_bundle(val_dataset, list(val_indices), randomize=True)

    ctrl_dataset = AlbumentationsDataset(dataset, TRANSFORMS_BUNDLE.eval)
    ctrl_dataset = TransformDataset(ctrl_dataset, TRANSFORMS_BUNDLE.model)
    ctrl_bundle = prepare_eval_bundle(ctrl_dataset, list(ctrl_indices), randomize=False)

    return ModelFitBundle(train1=train_bundle1,
                          train2=train_bundle2,
                          val=val_bundle,
                          ctrl=ctrl_bundle)


def fit_model(dataset_config: Dict[str, str],
              experiment_config: ExperimentConfig,
              device: Union[torch.device, int],
              tracker: Tracker,
              debug: bool,
              display_progress: bool) -> Dict[str, float]:
    # Initialization
    num_classes = len(TARGET_CLASSES)

    # Data loading
    data_bundle = prepare_model_fit_bundle(dataset_config, experiment_config.mixup)

    # Adjusting model topology
    model = torch.hub.load(experiment_config.repo, experiment_config.network, pretrained=True, verbose=False)

    old_fc = model.fc
    model.fc = nn.Linear(in_features=old_fc.in_features, out_features=num_classes)

    target_probs = target_distribution(data_bundle.train1.loader, classes=num_classes, display_progress=display_progress)
    init_softmax_logits(model.fc.bias, target_probs)

    model.type(torch.cuda.FloatTensor)
    model.to(device)

    # Preparing parameters
    param_groups = {
        'A': ParamGroup({'lr': experiment_config.hparams['lrA'], 'weight_decay': experiment_config.hparams['wdA']}),
        'A_no_decay': ParamGroup({'lr': experiment_config.hparams['lrA'], 'weight_decay': 0}),
        'B': ParamGroup({'lr': experiment_config.hparams['lrB'], 'weight_decay': experiment_config.hparams['wdB']}),
        'B_no_decay': ParamGroup({'lr': experiment_config.hparams['lrB'], 'weight_decay': 0})
    }

    def add_param(group_name: str, module_name: str, param_name: str, param: Parameter):
        if param.requires_grad:
            group = param_groups[group_name]
            group.params.append((f'{module_name}.{param_name}', param))

    def freeze_params():
        for index, (name, child) in enumerate(model.named_children()):
            if index < experiment_config.freeze:
                if debug:
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
        if debug:
            for group_name, group in param_groups.items():
                print(f'{group_name}: {group}')

    def prepare_optimizer_params():
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

    # Setting up optimizer/loss/scheduler/data retriever
    prepared_group_names, prepared_optimizer_params = prepare_optimizer_params()
    optimizer = optim.AdamW(prepared_optimizer_params)
    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=int(round(experiment_config.hparams['cos_t0'])),
                                                                    T_mult=int(round(experiment_config.hparams['cos_f'])))
    warmup_scheduler = WarmupScheduler(optimizer, epochs=3, next_scheduler=main_scheduler)

    if experiment_config.mixup:
        train_retriever = MixupDataRetriever(retriever1=DirectDataRetriever(data_bundle.train1.loader, target_format=TargetFormat.CLASS_INDEX),
                                             retriever2=DirectDataRetriever(data_bundle.train2.loader, target_format=TargetFormat.CLASS_INDEX),
                                             num_classes=num_classes,
                                             alpha=experiment_config.hparams['alpha'])
        train_accuracy = JaccardAccuracy()
    else:
        train_retriever = DirectDataRetriever(loader=data_bundle.train1.loader, target_format=TargetFormat.CLASS_INDEX)
        train_accuracy = ExactMatchAccuracy()

    smooth_dist = torch.full([num_classes], fill_value=1.0 / num_classes)
    train_loss = SmoothedCrossEntropyLoss(smooth_factor=experiment_config.hparams['smooth'],
                                          smooth_dist=smooth_dist,
                                          target_format=train_retriever.target_format)

    eval_accuracy = ExactMatchAccuracy()
    eval_loss = CrossEntropyLoss()

    # Training setup
    trial_tracker = tracker.new_trial(experiment_config.hparams)
    fit(device,
        model,
        TARGET_CLASSES,
        train_retriever=train_retriever,
        val_retriever=DirectDataRetriever(data_bundle.val.loader, target_format=TargetFormat.CLASS_INDEX),
        ctrl_retriever=DirectDataRetriever(data_bundle.ctrl.loader, target_format=TargetFormat.CLASS_INDEX),
        train_loss=train_loss,
        eval_loss=eval_loss,
        train_accuracy=train_accuracy,
        eval_accuracy=eval_accuracy,
        optimizer=optimizer,
        scheduler=warmup_scheduler,
        group_names=prepared_group_names,
        num_epochs=experiment_config.max_epochs,
        tracker=trial_tracker,
        display_progress=display_progress,
        sampled_epochs=experiment_config.max_epochs)

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


def score_model_blend(dataset_config: Dict[str, str],
                      device: Union[torch.device, int],
                      paths: List[Tuple[str, str]]):
    test_bundle = prepare_model_test_bundle(dataset_config)

    models = []
    eval_loss = None
    for directory, snapshot in paths:
        snapshot = Snapshotter.load_snapshot(directory, snapshot)
        model = snapshot.model.cpu()

        models.append(model)
        eval_loss = snapshot.eval_loss

        del snapshot

    score_blend(device, models, DirectDataRetriever(test_bundle.loader, target_format=TargetFormat.CLASS_INDEX), eval_loss)


def target_distribution(loader: DataLoader, classes: int, display_progress: bool = True) -> torch.Tensor:
    batches_count = math.ceil(len(loader.sampler) / loader.batch_size)

    with torch.no_grad():
        counts = torch.zeros(classes)
        with tqdm(total=batches_count, disable=not display_progress) as progress_bar:
            for _, y, _ in loader:
                counts.index_add_(0, y, torch.ones(y.shape[0]))
                progress_bar.update()

            total = float(counts.sum())
            probs = torch.div(counts, total)

        if display_progress:
            print('classes = [' + ', '.join([f'{x:.3f}' for x in probs.numpy().tolist()]) + ']')

        return probs
