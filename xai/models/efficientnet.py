from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.model import load_from_checkpoint


class EfficientNet(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        in_features = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = nn.Linear(in_features, num_classes)

        self.efficientnet.to(DEVICE)

        if checkpoint_path:
            load_from_checkpoint(self.efficientnet, checkpoint_path)

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

    def prepare_for_finetuning(self):
        # freeze all params
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # unfreeze conv layers
        for layer in self.get_layers().values():
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze classification layer
        for param in self.efficientnet.classifier[-1].parameters():
            param.requires_grad = True

    def get_model(self) -> torch.nn.Module:
        return self.efficientnet
