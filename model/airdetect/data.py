import os
from dataclasses import dataclass
from typing import Callable, List, Dict
from typing import Optional

import pandas as pd
from PIL.Image import Image as Img
from dacite import from_dict

from wheel5.dataset import LMDBImageDataset, SimpleImageClassificationDataset
from wheel5.dataset import SimpleImageDetectionDataset

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


@dataclass
class ClassifierDatasetConfig:
    image_dir: str
    metadata: Optional[str]
    annotations: Optional[str]
    lmdb_dir: Optional[str]

    @classmethod
    def from_dict(cls, d: Dict[str, str]):
        return from_dict(cls, d)


@dataclass
class DetectorDatasetConfig:
    image_dir: str
    metadata: Optional[str]

    @classmethod
    def from_dict(cls, d: Dict[str, str]):
        return from_dict(cls, d)


def load_classes(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if line != '']


def reverse_classes(classes: List[str]) -> Dict[str, int]:
    classes_to_num = {v: k for k, v in enumerate(classes)}
    assert len(classes) == len(classes_to_num)

    return classes_to_num


def coco_categories_unique() -> List[str]:
    na_counter = 0
    categories = []

    for category in COCO_INSTANCE_CATEGORY_NAMES:
        if category == 'N/A':
            category = f'{category}_{na_counter}'
            na_counter += 1

        categories.append(category)

    return categories


def load_classifier_dataset(config: ClassifierDatasetConfig,
                            target_classes: List[str],
                            transform: Callable[[Img], Img] = None,
                            name: str = ''):

    if config.metadata is None:
        entries = []
        for filename in os.listdir(config.image_dir):
            entries.append({'path': filename, 'target': -1})
        df_metadata = pd.DataFrame(entries)
    else:
        df_metadata = pd.read_csv(filepath_or_buffer=config.metadata, sep=',', header=0)

        classes_map = {cls: i for i, cls in enumerate(target_classes)}
        df_metadata['target'] = df_metadata['name'].map(lambda class_name: classes_map.get(class_name))
        df_metadata = df_metadata[df_metadata['target'].notnull()]
        df_metadata['target'] = df_metadata['target'].astype('int64')

    if config.lmdb_dir is None:
        return SimpleImageClassificationDataset(df_metadata,
                                                image_dir=config.image_dir,
                                                transform=transform,
                                                name=name)
    else:
        return LMDBImageDataset.cached(df_metadata,
                                       image_dir=config.image_dir,
                                       lmdb_path=config.lmdb_dir,
                                       lmdb_map_size=(1024**4),
                                       transform=transform,
                                       name=name)


def load_detector_dataset(config: DetectorDatasetConfig, name: str = ''):
    entries = []
    for filename in os.listdir(config.image_dir):
        entries.append({'path': filename})
    df_metadata = pd.DataFrame(entries)

    return SimpleImageDetectionDataset(df_metadata,
                                       image_dir=config.image_dir,
                                       name=name)
