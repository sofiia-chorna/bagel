from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision.models import VGG16_Weights, vgg16

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.model import load_from_checkpoint


class VGG16(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)

        in_features = int(self.vgg16.classifier[-1].in_features)  # type: ignore
        self.vgg16.classifier[-1] = torch.nn.Linear(in_features, num_classes)

        self.vgg16.to(DEVICE)

        if checkpoint_path:
            load_from_checkpoint(self.vgg16, checkpoint_path)

    def forward(self, x: Tensor):
        return self.vgg16(x)  # type: ignore

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv1_1": self.vgg16.features[0],
            "conv2_1": self.vgg16.features[5],
            "conv3_1": self.vgg16.features[10],
            "conv4_1": self.vgg16.features[17],
            "conv5_1": self.vgg16.features[24],
        }

    def get_name(self) -> str:
        return self.vgg16._get_name()

    def prepare_for_finetuning(self):
        # freeze all params
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # unfreeze conv layers
        for layer in self.get_layers().values():
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze classification layer
        for param in self.vgg16.classifier[-1].parameters():
            param.requires_grad = True

    def get_model(self) -> torch.nn.Module:
        return self.vgg16
