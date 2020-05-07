import logging
import os
import pathlib
import sys
from dataclasses import asdict
from typing import Callable, Optional, List
from typing import Dict

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as Img
from matplotlib.figure import Figure
from torch.utils.data import Subset
from tqdm import tqdm
import albumentations as albu

from wheel5.storage import LMDBDict
from wheel5.tasks.detection import convert_bboxes, filter_bboxes, non_maximum_suppression, Rectangle
from wheel5.util import shape
from wheel5.viz.predictions import draw_bboxes
from wheel5.storage import BoundingBoxesLMDBDict
import wheel5.transforms.albumentations as albu_ext
from .detector import AircraftDetector, AircraftDetectorConfig
from ..data import DetectorDatasetConfig, load_detector_dataset

Transform = Callable[[Img], Img]


def visualize_predicted_bboxes(dataset_config: Dict[str, str],
                               device: int,
                               samples: int,
                               min_score: float = 0.0,
                               top_bboxes: Optional[int] = None,
                               categories: Optional[List[str]] = None,
                               directory: Optional[str] = None) -> Figure:
    device = torch.device(f'cuda:{device}')

    config = AircraftDetectorConfig(random_state_seed=42)
    hparams = asdict(config)

    model = AircraftDetector(hparams)
    model.to(device)
    model.freeze()
    model.eval()

    categories_num = [model.categories_to_num[category] for category in categories]

    dataset = model.load_dataset(config=DetectorDatasetConfig.from_dict(dataset_config), name='main')
    loader = model.prepare_eval_loader(dataset, name='main')

    x_list = []
    bboxes_list = []
    count = 0

    with torch.no_grad():
        for x, _, indices, *_ in loader:
            x_device = [t.to(device) for t in x]
            z = model(x_device)

            for i in range(0, len(indices)):
                z_i_cpu = {k: v.cpu() for k, v in z[i].items()}

                bboxes = convert_bboxes(boxes=z_i_cpu['boxes'], labels=z_i_cpu['labels'], scores=z_i_cpu['scores'])
                bboxes = filter_bboxes(bboxes, min_score=min_score, top_bboxes=top_bboxes, categories=categories_num)

                x_list.append(x[i])
                bboxes_list.append(bboxes)

            count += len(x)
            if count >= samples:
                break

    x = x_list[0:samples]
    bboxes = bboxes_list[0:samples]

    draw_bboxes(x, bboxes=bboxes, categories=model.categories, directory=directory)


def visualize_stored_bboxes(dataset_config: Dict[str, str],
                            bboxes_path: str,
                            samples: int,
                            randomize: bool = False,
                            min_score: float = 0.0,
                            top_bboxes: Optional[int] = None,
                            nms_threshold: Optional[float] = None,
                            nms_ranking: str = 'score_sqrt_area',
                            nms_suppression: str = 'overlap',
                            expand_coeff: float = 0.0,
                            directory: Optional[str] = None) -> Figure:
    config = AircraftDetectorConfig(random_state_seed=42)
    hparams = asdict(config)

    model = AircraftDetector(hparams)
    model.freeze()
    model.eval()

    dataset = model.load_dataset(config=DetectorDatasetConfig.from_dict(dataset_config), name='main')
    subset_indices = list(range(0, len(dataset)))

    if randomize:
        np.random.shuffle(subset_indices)

    dataset = Subset(dataset=dataset, indices=subset_indices[0:samples])
    loader = model.prepare_eval_loader(dataset, name='main')

    x_list = []
    bboxes_list = []

    with LMDBDict(bboxes_path) as lmdb_dict:
        bboxes_db = BoundingBoxesLMDBDict(lmdb_dict)

        with torch.no_grad():
            for x, _, _, paths, *_ in loader:
                for i in range(0, len(paths)):
                    bboxes = bboxes_db[paths[i]]
                    bboxes = filter_bboxes(bboxes, min_score=min_score, top_bboxes=top_bboxes)

                    if nms_threshold is not None:
                        bboxes = non_maximum_suppression(bboxes, nms_threshold, nms_ranking, nms_suppression)

                    _, h, w = x[i].shape
                    x_i_rect = Rectangle(pt_from=(0, 0), pt_to=(w - 1, h - 1))
                    bboxes = [bbox.expand(expand_coeff).intersection(x_i_rect) for bbox in bboxes]

                    x_list.append(x[i])
                    bboxes_list.append(bboxes)

    draw_bboxes(x_list, bboxes=bboxes_list, categories=model.categories, directory=directory)


def find_empty_bboxes(bboxes_path: str) -> List[str]:
    bad_paths = []

    with LMDBDict(bboxes_path) as lmdb_dict:
        bboxes_db = BoundingBoxesLMDBDict(lmdb_dict)
        for path, bboxes in bboxes_db.items():
            if len(bboxes) == 0:
                bad_paths.append(path)

    return bad_paths


def build_bboxes(dataset_config: Dict[str, str],
                 bboxes_path: str,
                 device: int,
                 show_progress: bool = True):
    device = torch.device(f'cuda:{device}')

    config = AircraftDetectorConfig(random_state_seed=42)
    hparams = asdict(config)

    model = AircraftDetector(hparams)
    model.to(device)
    model.freeze()
    model.eval()

    airplane_categories = [model.categories_to_num['airplane']]

    dataset = load_detector_dataset(config=DetectorDatasetConfig.from_dict(dataset_config), name='main')
    loader = model.prepare_eval_loader(dataset, name='main')

    logger = logging.getLogger(f'{__name__}')

    with LMDBDict(bboxes_path) as lmdb_dict:
        bboxes_db = BoundingBoxesLMDBDict(lmdb_dict)

        with tqdm(total=len(loader), disable=not show_progress, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f'Building bboxes')

            for x, _, indices, paths, *_ in loader:
                x = [t.to(device) for t in x]
                z = model(x)

                for i in range(0, len(indices)):
                    z_i_cpu = {k: v.cpu() for k, v in z[i].items()}

                    bboxes = convert_bboxes(boxes=z_i_cpu['boxes'], labels=z_i_cpu['labels'], scores=z_i_cpu['scores'])
                    bboxes = filter_bboxes(bboxes, categories=airplane_categories)

                    bboxes_db[paths[i]] = bboxes
                    logger.info(f'bounding boxes: #{paths[i]} - {shape(x[i])}:\n' + '\n'.join([f'    {str(bbox)}' for bbox in bboxes]))

                progress_bar.update()


def crop_by_bboxes(image_dir_in,
                   image_dir_out,
                   bboxes_path: str,
                   expand_coeff: float = 0.25,
                   min_score: float = 0.7,
                   nms_threshold: float = 0.5,
                   nms_ranking: str = 'score_sqrt_area',
                   nms_suppression: str = 'overlap',
                   show_progress: bool = True):

    pathlib.Path(image_dir_out).mkdir(parents=True, exist_ok=False)

    paths = list(os.listdir(image_dir_in))

    with LMDBDict(bboxes_path) as lmdb_dict:
        bboxes_db = BoundingBoxesLMDBDict(lmdb_dict)

        with tqdm(total=len(paths), disable=not show_progress, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f'Cropping images {image_dir_in} => {image_dir_out}')

            for path in paths:
                img = Image.open(os.path.join(image_dir_in, path))

                w, h = img.size
                img_rect = Rectangle(pt_from=(0, 0), pt_to=(w - 1, h - 1))

                bboxes = bboxes_db[path]
                bboxes = filter_bboxes(bboxes, min_score=min_score)
                bboxes = non_maximum_suppression(bboxes, nms_threshold, nms_ranking, nms_suppression)

                if len(bboxes) > 0:
                    bbox = bboxes[0]
                    bbox = bbox.expand(expand_coeff)
                    bbox = bbox.intersection(img_rect)

                    transform = albu.Crop(x_min=bbox.x0, y_min=bbox.y0, x_max=bbox.x1, y_max=bbox.y1, always_apply=True)
                else:
                    transform = albu_ext.Identity()

                augmented = transform(image=np.array(img))
                img_aug = Image.fromarray(augmented['image'])

                img_aug.save(os.path.join(image_dir_out, path))

                progress_bar.update()
