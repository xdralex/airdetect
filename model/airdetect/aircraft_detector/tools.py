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
        for x, _, *_ in loader:
            x = [t.to(device) for t in x]
            z = model.forward(x)

            x_list += x
            z_list += z

            count += len(x)
            if count >= samples:
                break

    x = x_list[0:samples]
    z = z_list[0:samples]

    print(z)
