from typing import Optional

import torch
from torch import Tensor
from torchvision.models import AlexNet_Weights, alexnet

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.model import load_from_checkpoint


class AlexNet(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)

        in_features = int(self.alexnet.classifier[6].in_features)  # type: ignore
        self.alexnet.classifier[6] = torch.nn.Linear(in_features, num_classes)

        self.alexnet.to(DEVICE)

        if checkpoint_path:
            load_from_checkpoint(self.alexnet, checkpoint_path)

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

    def prepare_for_finetuning(self):
        # freeze all params
        for param in self.alexnet.parameters():
            param.requires_grad = False

        # unfreeze conv layers
        for layer in self.get_layers().values():
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze classification layer
        for param in self.alexnet.classifier[6].parameters():
            param.requires_grad = True

    def get_model(self) -> torch.nn.Module:
        return self.alexnet
