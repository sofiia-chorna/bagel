from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE


class EfficientNet(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        in_features = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = nn.Linear(in_features, num_classes)

        self.efficientnet.to(DEVICE)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.efficientnet.load_state_dict(checkpoint, strict=False)

    def forward(self, x: Tensor):
        return self.efficientnet(x)

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv_stem": self.efficientnet.features[0],
            "block1": self.efficientnet.features[1][-1],
            "block3": self.efficientnet.features[3][-1],
            "block5": self.efficientnet.features[5][-1],
            "conv_head": self.efficientnet.features[8],
        }

    def get_name(self) -> str:
        return self.efficientnet._get_name()
