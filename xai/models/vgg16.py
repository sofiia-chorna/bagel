from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision import models

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE


class VGG16(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        in_features = int(self.vgg16.classifier[-1].in_features)  # type: ignore
        self.vgg16.classifier[-1] = torch.nn.Linear(in_features, num_classes)

        self.vgg16.to(DEVICE)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.vgg16.load_state_dict(checkpoint, strict=False)

    def forward(self, x: Tensor):
        return self.vgg16(x)  # type: ignore

    def get_conv_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv1_1": self.vgg16.features[0],
            "conv2_1": self.vgg16.features[5],
            "conv3_1": self.vgg16.features[10],
            "conv4_1": self.vgg16.features[17],
            "conv5_1": self.vgg16.features[24],
        }

    def get_name(self) -> str:
        return self.vgg16._get_name()
