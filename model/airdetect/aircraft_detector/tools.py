import logging
from dataclasses import asdict
from typing import Callable, Optional, List
from typing import Dict

import numpy as np
import torch
from PIL.Image import Image as Img
from matplotlib.figure import Figure
from torch.utils.data import Subset
from tqdm import tqdm

from wheel5.storage import LMDBDict, encode_list, decode_list
from wheel5.tasks.detection import BoundingBox, convert_bboxes, filter_bboxes, non_maximum_suppression
from wheel5.util import shape
from wheel5.viz.predictions import draw_bboxes
from .detector import AircraftDetector, AircraftDetectorConfig
from ..data import DetectorDatasetConfig, load_detector_dataset

Transform = Callable[[Img], Img]


def visualize_predictions(dataset_config: Dict[str, str],
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
        for x, _, indices in loader:
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


def visualize_stored_predictions(dataset_config: Dict[str, str],
                                 bboxes_path: str,
                                 samples: int,
                                 randomize: bool = False,
                                 min_score: float = 0.0,
                                 top_bboxes: Optional[int] = None,
                                 nms_threshold: Optional[float] = None,
                                 nms_ranking: str = 'score_sqrt_area',
                                 nms_suppression: str = 'overlap',
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

    with LMDBDict(bboxes_path) as bboxes_db:
        with torch.no_grad():
            for x, _, indices in loader:
                for i in range(0, len(indices)):
                    key = str(int(indices[i]))

                    bboxes = [BoundingBox.decode(b) for b in decode_list(bboxes_db[key], size=BoundingBox.byte_size())]
                    bboxes = filter_bboxes(bboxes, min_score=min_score, top_bboxes=top_bboxes)

                    if nms_threshold is not None:
                        bboxes = non_maximum_suppression(bboxes, nms_threshold, nms_ranking, nms_suppression)

                    x_list.append(x[i])
                    bboxes_list.append(bboxes)

    draw_bboxes(x_list, bboxes=bboxes_list, categories=model.categories, directory=directory)


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

    with LMDBDict(bboxes_path) as bboxes_db:
        with tqdm(total=len(loader), disable=not show_progress) as progress_bar:
            progress_bar.set_description(f'Building bboxes')

            for x, _, indices in loader:
                x = [t.to(device) for t in x]
                z = model(x)

                for i in range(0, len(indices)):
                    z_i_cpu = {k: v.cpu() for k, v in z[i].items()}

                    bboxes = convert_bboxes(boxes=z_i_cpu['boxes'], labels=z_i_cpu['labels'], scores=z_i_cpu['scores'])
                    bboxes = filter_bboxes(bboxes, categories=airplane_categories)

                    key = str(int(indices[i]))
                    bboxes_db[key] = encode_list([bbox.encode() for bbox in bboxes], size=BoundingBox.byte_size())

                    logger.info(f'#{key} - {shape(x[i])}:\n' + '\n'.join([f'    {str(bbox)}' for bbox in bboxes]))

                progress_bar.update()
