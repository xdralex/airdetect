import click
import yaml

from wheel5 import logutils
from .tools import build_bboxes, crop_by_bboxes


@click.option('-d', '--device', 'device', default='0', help='device number (0, 1, ...)', type=int)
@click.option('-s', '--dataset', 'dataset', required=True, help='dataset name', type=str)
def cli_build_bboxes(device: int, dataset: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    build_bboxes(dataset_config=config['datasets'][dataset],
                 bboxes_path=config['boost']['bboxes'][dataset],
                 device=device)


@click.option('-s', '--dataset', 'dataset', required=True, help='dataset name', type=str)
def cli_crop_by_bboxes(dataset: str):
    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
        logutils.configure_logging(config['logging'])

    dataset_config = config['datasets'][dataset]

    image_dir_in = dataset_config['image_dir']
    image_dir_out = f'{image_dir_in}_crop'
    bboxes_path = config['boost']['bboxes'][dataset]

    crop_by_bboxes(image_dir_in, image_dir_out, bboxes_path)
