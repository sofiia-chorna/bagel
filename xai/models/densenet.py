from typing import Dict, Literal, Optional

import torch
from torch import Tensor, nn
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet169_Weights,
    densenet121,
    densenet169,
)

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE


class DenseNet(BaseModel):
    def __init__(
        self,
        num_classes: int,
        type: Literal["densenet121", "densenet169"] = "densenet121",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        match type:
            case "densenet121":
                self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
            case "densenet169":
                self.densenet = densenet169(weights=DenseNet169_Weights.DEFAULT)
            case _:
                raise ValueError(f"Unknown type of DenseNet: {type}")

        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(in_features, num_classes)

        self.densenet.to(DEVICE)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.densenet.load_state_dict(checkpoint, strict=False)

    def forward(self, x: Tensor):
        return self.densenet(x)  # type: ignore

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv0": self.densenet.features.conv0,  # type: ignore
            "denseblock1": self.densenet.features.denseblock1[-1],  # type: ignore
            "denseblock2": self.densenet.features.denseblock2[-1],  # type: ignore
            "denseblock3": self.densenet.features.denseblock3[-1],  # type: ignore
            "denseblock4": self.densenet.features.denseblock4[-1],  # type: ignore
        }

    def get_name(self) -> str:
        return self.densenet._get_name()
