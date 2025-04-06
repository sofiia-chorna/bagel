from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision.models import GoogLeNet_Weights, googlenet

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.model import load_from_checkpoint


class GoogLeNet(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.googlenet = googlenet(weights=GoogLeNet_Weights.DEFAULT)

        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)

        self.googlenet.to(DEVICE)

        if checkpoint_path:
            load_from_checkpoint(self.googlenet, checkpoint_path)

    def forward(self, x: Tensor):
        return self.googlenet(x)

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv1": self.googlenet.conv1,
            "inception3a": self.googlenet.inception3a,
            "inception4a": self.googlenet.inception4a,
            "inception4e": self.googlenet.inception4e,
            "inception5b": self.googlenet.inception5b,
        }

    def get_name(self) -> str:
        return "GoogLeNet"

    def prepare_for_finetuning(self):
        # freeze all params
        for param in self.googlenet.parameters():
            param.requires_grad = False

        # unfreeze conv layers
        for layer in self.get_layers().values():
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze classification layer
        for param in self.googlenet.fc.parameters():
            param.requires_grad = True

    def get_model(self) -> torch.nn.Module:
        return self.googlenet
