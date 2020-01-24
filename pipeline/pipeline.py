import math

import hyperopt
import numpy as np
import torch
import yaml
from hyperopt import hp, fmin
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from wheel5 import logutils
from wheel5.dataset import LMDBImageDataset, WrappingTransformDataset, split_indices
from wheel5.model import fit
from wheel5.organizer import Organizer
from wheel5.snapshotters import CheckpointSnapshotter, BestCVSnapshotter
from wheel5.transforms import SquarePaddedResize

from data import load_aircraft_data


random_state = np.random.RandomState(42)


with open('config.yaml', 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

    logutils.configure_logging(config['logging'])

    snapshot_root = config['paths']['snapshot_root']
    tensorboard_root = config['paths']['tensorboard_root']
    lmdb_dir = config['paths']['lmdb_dir']
    dataset_dir = config['paths']['dataset_dir']

df_images = load_aircraft_data(config['db'], 'aircraft_photos_snapshot', random_state)


lmdb_dataset = LMDBImageDataset.cached(df_images,
                                       image_dir=dataset_dir,
                                       lmdb_path=lmdb_dir,
                                       prepare_transform=SquarePaddedResize(size=224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def transform_fn(arg):
    image, cls, name, index = arg
    return transform(image), cls, name, index


dataset = WrappingTransformDataset(
    wrapped=lmdb_dataset,
    transform_fn=transform_fn
)


indices = list(range(len(dataset)))

test_indices, nontest_indices = split_indices(indices, split=0.2, random_state=random_state)

pub_test_indices, priv_test_indices = split_indices(test_indices, split=0.5, random_state=random_state)
aux_indices, train_indices = split_indices(nontest_indices, split=0.3, random_state=random_state)
stack_indices, val_indices = split_indices(aux_indices, split=0.333, random_state=random_state)

pub_test_sampler = SubsetRandomSampler(pub_test_indices)
priv_test_sampler = SubsetRandomSampler(priv_test_indices)

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
stack_sampler = SubsetRandomSampler(stack_indices)


pub_test_loader = DataLoader(dataset, batch_size=256, sampler=pub_test_sampler, num_workers=4, pin_memory=True)
priv_test_loader = DataLoader(dataset, batch_size=256, sampler=priv_test_sampler, num_workers=4, pin_memory=True)

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(dataset, batch_size=256, sampler=val_sampler, num_workers=4, pin_memory=True)
stack_loader = DataLoader(dataset, batch_size=256, sampler=stack_sampler, num_workers=4, pin_memory=True)


device = torch.device('cuda:0')


org = Organizer(snapshot_root, tensorboard_root, experiment='resnet18_hp4')


def fit_trial(config):
    # Model preparation
    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet18', pretrained=True, verbose=False)

    old_fc = model.fc
    model.fc = nn.Linear(in_features=old_fc.in_features, out_features=dataset.wrapped.classes())

    model.type(torch.cuda.FloatTensor)
    model.to(device)

    loss = nn.CrossEntropyLoss()

    params_old = list([param for name, param in model.named_parameters() if not name.startswith('fc.')])
    params_new = list([param for name, param in model.named_parameters() if name.startswith('fc.')])
    optimizer_params = [
        {'params': params_old, 'lr': config['lrA'], 'weight_decay': config['wdA']},
        {'params': params_new, 'lr': config['lrB'], 'weight_decay': config['wdB']}
    ]
    optimizer = optim.AdamW(optimizer_params)

    # Training setup
    org_trial = org.new_trial(hparams=config)
    snapshot_dir = org_trial.snapshot_dir()
    tensorboard_dir = org_trial.tensorboard_dir()

    tb_writer = SummaryWriter(tensorboard_dir, max_queue=100, flush_secs=60)

    metrics_df = fit(device, model, train_loader, val_loader, loss, optimizer,
                     num_epochs=20,
                     snapshotter=[
                         CheckpointSnapshotter(snapshot_dir, frequency=10),
                         BestCVSnapshotter(snapshot_dir, metric_name='accuracy', asc=False, best=3),
                         BestCVSnapshotter(snapshot_dir, metric_name='loss', asc=True, best=3),
                     ],
                     tb_writer=tb_writer,
                     display_progress=False)

    # Reporting
    results = {
        'hp/best_val_acc': metrics_df['val_accuracy'].max(),
        'hp/best_val_loss': metrics_df['val_loss'].min(),
        'hp/final_val_acc': metrics_df['val_accuracy'].iloc[-1],
        'hp/final_val_loss': metrics_df['val_loss'].iloc[-1],

        'hp/best_train_acc': metrics_df['train_accuracy'].max(),
        'hp/best_train_loss': metrics_df['train_loss'].min(),
        'hp/final_train_acc': metrics_df['train_accuracy'].iloc[-1],
        'hp/final_train_loss': metrics_df['train_loss'].iloc[-1],
    }

    tb_writer.add_hparams(config, results)
    tb_writer.flush()

    return results['hp/best_val_loss']


space = {
    'lrA': hp.loguniform('lrA', math.log(1e-5), math.log(1)),
    'wdA': hp.loguniform('wdA', math.log(1e-4), math.log(1)),
    'lrB': hp.loguniform('lrB', math.log(1e-5), math.log(1)),
    'wdB': hp.loguniform('wdB', math.log(1e-4), math.log(1)),
}

best = fmin(fit_trial, space=space, algo=hyperopt.rand.suggest, max_evals=30)
