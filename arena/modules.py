import torch
from torch import nn


class ResnetModel(nn.Module):
    def __init__(self, hub: str, name: str, num_classes: int):
        super(ResnetModel, self).__init__()

        self.model = torch.hub.load(hub, name, pretrained=True, verbose=False)

        old_fc = self.model.fc
        self.model.fc = nn.Linear(in_features=old_fc.in_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
