from typing import Dict, Literal, Optional

import torch
from torch import Tensor, nn
from torchvision.models import (
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
    resnext50_32x4d,
    resnext101_32x8d,
)

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.model import load_from_checkpoint


class ResNeXt(BaseModel):
    def __init__(
        self,
        num_classes: int,
        type: Literal["resnext50", "resnext101"] = "resnext50",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.type = type  # type: ignore

        match type:
            case "resnext50":
                self.resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            case "resnext101":
                self.resnext = resnext101_32x8d(
                    weights=ResNeXt101_32X8D_Weights.DEFAULT
                )
            case _:
                raise ValueError(f"Unknown type of ResNeXt: {type}")

        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)

        self.resnext.to(DEVICE)

        if checkpoint_path:
            load_from_checkpoint(self.resnext, checkpoint_path)

    def forward(self, x: Tensor):
        return self.resnext(x)

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "conv1": self.resnext.conv1,
            "layer1": self.resnext.layer1[-1],
            "layer2": self.resnext.layer2[-1],
            "layer3": self.resnext.layer3[-1],
            "layer4": self.resnext.layer4[-1],
        }

    def get_name(self) -> str:
        match self.type:
            case "resnext50":
                return "ResNext50"
            case "resnext101":
                return "ResNext101"
            case _:
                return self.resnext._get_name()

    def prepare_for_finetuning(self):
        # freeze all params
        for param in self.resnext.parameters():
            param.requires_grad = False

        # unfreeze conv layers
        for layer in self.get_layers().values():
            for param in layer.parameters():
                param.requires_grad = True

        # unfreeze classification layer
        for param in self.resnext.fc.parameters():
            param.requires_grad = True

    def get_model(self) -> torch.nn.Module:
        return self.resnext
