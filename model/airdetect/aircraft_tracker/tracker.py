from dataclasses import dataclass
from typing import List, Dict

import albumentations as albu
import cv2
import torch
from torch import nn
from torch.nn.functional import log_softmax
from torchvision import transforms

import wheel5.transforms.albumentations as albu_ext
from airdetect.aircraft_classifier import AircraftClassifier
from airdetect.aircraft_detector import AircraftDetector
from wheel5.dataset import AlbumentationsTransform
from wheel5.tasks.detection import convert_bboxes, filter_bboxes, non_maximum_suppression, Rectangle, BoundingBox


@dataclass
class ClassifiedBBox:
    frame: BoundingBox
    classes: Dict[str, float]


class AircraftTracker(nn.Module):
    def __init__(self,
                 detector: AircraftDetector,
                 classifier: AircraftClassifier,
                 expand_coeff: float = 0.25,
                 min_score: float = 0.7,
                 top_bboxes: int = 8,
                 nms_threshold: float = 0.5,
                 nms_ranking: str = 'score_sqrt_area',
                 nms_suppression: str = 'overlap'):

        super(AircraftTracker, self).__init__()

        self.detector = detector
        self.classifier = classifier

        self.expand_coeff = expand_coeff
        self.min_score = min_score
        self.top_bboxes = top_bboxes
        self.nms_threshold = nms_threshold
        self.nms_ranking = nms_ranking
        self.nms_suppression = nms_suppression

        self.airplane_categories = [detector.categories_to_num['airplane']]

        self.image_to_tensor = transforms.ToTensor()
        self.tensor_to_image = transforms.ToPILImage()

        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]
        mean_color = tuple([int(round(c * 255)) for c in normalize_mean])

        self.classifier_transform = transforms.Compose([
            AlbumentationsTransform(albu.Compose([
                albu_ext.PadToSquare(fill=mean_color),
                albu.Resize(height=224, width=224, interpolation=cv2.INTER_AREA)
            ])),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])

    def forward(self, x: List[torch.Tensor]) -> List[List[ClassifiedBBox]]:
        results = []

        bx_raw = self.detector(x)
        for i in range(len(x)):
            x_i = x[i].cpu()
            bx_raw_i = {k: v.cpu() for k, v in bx_raw[i].items()}

            bboxes = convert_bboxes(boxes=bx_raw_i['boxes'], labels=bx_raw_i['labels'], scores=bx_raw_i['scores'])
            bboxes = filter_bboxes(bboxes, min_score=self.min_score, categories=self.airplane_categories)
            bboxes = non_maximum_suppression(bboxes, self.nms_threshold, self.nms_ranking, self.nms_suppression)
            bboxes = bboxes[:self.top_bboxes]

            image = self.tensor_to_image(x_i)
            w, h = image.size
            image_rect = Rectangle(pt_from=(0, 0), pt_to=(w - 1, h - 1))

            classified_bboxes = []
            if len(bboxes) > 0:
                cropped_tensors = []
                for bbox in bboxes:
                    bbox = bbox.expand(self.expand_coeff)
                    bbox = bbox.intersection(image_rect)

                    crop = AlbumentationsTransform(albu.Crop(x_min=bbox.x0, y_min=bbox.y0, x_max=bbox.x1, y_max=bbox.y1, always_apply=True))
                    cropped_image = crop(image)

                    cropped_tensor = self.classifier_transform(cropped_image)
                    cropped_tensors.append(cropped_tensor)

                crops_i = torch.stack(cropped_tensors).to(x[i])
                z_i = self.classifier(crops_i)
                probs_i_hat = torch.exp(log_softmax(z_i, dim=1)).cpu().numpy()

                for j in range(len(bboxes)):
                    classified_bboxes.append(ClassifiedBBox(
                        frame=bboxes[j],
                        classes=dict(zip(self.classifier.target_classes, probs_i_hat[j].tolist()))
                    ))

            results.append(classified_bboxes)

        return results
