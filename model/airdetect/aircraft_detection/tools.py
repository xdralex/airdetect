from dataclasses import asdict
from dataclasses import asdict
from typing import Callable
from typing import Dict

import torch
from PIL.Image import Image as Img
from matplotlib.figure import Figure

from .pipeline import AircraftDetector, AircraftDetectorConfig

Transform = Callable[[Img], Img]


def visualize_predictions(dataset_config: Dict[str, str],
                          device: int,
                          samples: int) -> Figure:
    device = torch.device(f'cuda:{device}')

    config = AircraftDetectorConfig(random_state_seed=42)
    hparams = asdict(config)

    model = AircraftDetector(hparams)
    model.to(device)
    model.freeze()
    model.eval()

    dataset = model.load_dataset(dataset_config)
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

    print(z)
