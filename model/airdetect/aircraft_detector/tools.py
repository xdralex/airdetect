from dataclasses import asdict
from typing import Callable, Optional, List
from typing import Dict

import torch
from PIL.Image import Image as Img
from matplotlib.figure import Figure

from wheel5.tasks.detection import extract_bboxes
from wheel5.viz.predictions import draw_bboxes
from .detector import AircraftDetector, AircraftDetectorConfig

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

    dataset = model.load_dataset(dataset_config)
    loader = model.prepare_eval_loader(dataset)

    x_list = []
    bboxes_list = []
    count = 0

    with torch.no_grad():
        for x, _, *_ in loader:
            x_device = [t.to(device) for t in x]
            z = model(x_device)

            x_list += [t.cpu() for t in x]
            bboxes_list += [extract_bboxes(t, min_score, top_bboxes, categories_num) for t in z]

            count += len(x)
            if count >= samples:
                break

    x = x_list[0:samples]
    bboxes = bboxes_list[0:samples]

    draw_bboxes(x, bboxes=bboxes, categories=model.categories, directory=directory)
