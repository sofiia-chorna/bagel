from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision.models import Inception_V3_Weights, inception_v3

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.model import load_from_checkpoint


class InceptionV3(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)

        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, num_classes)

        self.inception.to(DEVICE)

        if checkpoint_path:
            load_from_checkpoint(self.inception, checkpoint_path)

    def forward(self, x: Tensor):
        return self.inception(x)

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "Conv2d_1a_3x3": self.inception.Conv2d_1a_3x3,
            "Mixed_5b": self.inception.Mixed_5b,
            "Mixed_6a": self.inception.Mixed_6a,
            "Mixed_7a": self.inception.Mixed_7a,
            "Mixed_7c": self.inception.Mixed_7c,
        }

    def get_name(self) -> str:
        return "InceptionV3"

    def prepare_for_finetuning(self):
        # freeze all params
        for param in self.inception.parameters():
            param.requires_grad = False

        # unfreeze conv layers
        for layer in self.get_layers().values():
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze classification layer
        for param in self.inception.fc.parameters():
            param.requires_grad = True

    def get_model(self) -> torch.nn.Module:
        return self.inception
