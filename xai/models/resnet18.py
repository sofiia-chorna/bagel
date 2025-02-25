from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision import models

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE


class ResNet18(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = int(self.resnet18.fc.in_features)
        self.resnet18.fc = nn.Linear(in_features, num_classes)

        self.resnet18.to(DEVICE)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.resnet18.load_state_dict(checkpoint, strict=False)

    def forward(self, x: Tensor):
        return self.resnet18(x)  # type: ignore

    def get_conv_layers(self) -> Dict[str, nn.Module]:
        return {
            "bn1": self.resnet18.bn1,
            "layer1": self.resnet18.layer1[-1],
            "layer2": self.resnet18.layer2[-1],
            "layer3": self.resnet18.layer3[-1],
            "layer4": self.resnet18.layer4[-1],
        }

    def get_name(self) -> str:
        return self.resnet18._get_name()
