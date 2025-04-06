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
from xai.utils.model import load_from_checkpoint


class DenseNet(BaseModel):
    def __init__(
        self,
        num_classes: int,
        type: Literal["densenet121", "densenet169"] = "densenet121",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.type = type  # type: ignore

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
            load_from_checkpoint(self.densenet, checkpoint_path)

    def forward(self, x: Tensor):
        return self.densenet(x)  # type: ignore

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv0": self.densenet.features.conv0,  # type: ignore
            "denseblock1": self.densenet.features.denseblock1,
            "denseblock2": self.densenet.features.denseblock2,
            "denseblock3": self.densenet.features.denseblock3,
            "denseblock4": self.densenet.features.denseblock4,
        }

    def get_name(self) -> str:
        match self.type:
            case "densenet121":
                return "DenseNet121"
            case "densenet169":
                return "DenseNet169"
            case _:
                return self.densenet._get_name()

    def prepare_for_finetuning(self):
        # freeze all params
        for param in self.densenet.parameters():
            param.requires_grad = False

        # unfreeze conv layers
        for layer in self.get_layers().values():
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze classification layer
        for param in self.densenet.classifier.parameters():
            param.requires_grad = True

    def get_model(self) -> torch.nn.Module:
        return self.densenet
