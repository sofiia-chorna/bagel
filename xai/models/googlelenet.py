from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision.models import GoogLeNet_Weights, googlenet

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE


class GoogLeNet(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.googlenet = googlenet(weights=GoogLeNet_Weights.DEFAULT)

        in_features = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(in_features, num_classes)

        self.googlenet.to(DEVICE)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.googlenet.load_state_dict(checkpoint, strict=False)

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
