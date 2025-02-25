from typing import Optional

import torch
from torch import Tensor
from torchvision.models import AlexNet_Weights, alexnet

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE


class AlexNet(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)

        in_features = int(self.alexnet.classifier[6].in_features)  # type: ignore
        self.alexnet.classifier[6] = torch.nn.Linear(in_features, num_classes)

        self.alexnet.to(DEVICE)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.alexnet.load_state_dict(checkpoint, strict=False)

    def forward(self, x: Tensor):
        return self.alexnet(x)  # type: ignore

    def get_layers(self) -> dict:
        return {
            "conv1": self.alexnet.features[0],
            "conv2": self.alexnet.features[3],
            "conv3": self.alexnet.features[6],
            "conv4": self.alexnet.features[8],
            "conv5": self.alexnet.features[10],
        }

    def get_name(self) -> str:
        return self.alexnet._get_name()
