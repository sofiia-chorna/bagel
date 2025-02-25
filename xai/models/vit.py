from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torchvision.models import ViT_B_32_Weights, vit_b_32

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE


class ViT(BaseModel):
    def __init__(self, num_classes: int, checkpoint_path: Optional[str] = None) -> None:
        super().__init__()
        self.vit = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)

        in_features = int(self.vit.heads[0].in_features)  # type: ignore
        self.vit.heads = nn.Linear(in_features, num_classes)  # type: ignore

        self.vit.to(DEVICE)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.vit.load_state_dict(checkpoint, strict=False)

    def forward(self, x: Tensor):
        return self.vit(x)  # type: ignore

    def get_layers(self) -> Dict[str, nn.Module]:
        return {
            "patch_embedding": self.vit.conv_proj,
            "encoder_layers": self.vit.encoder,
            "attention_layers": self.vit.encoder.layers,
        }

    def get_name(self) -> str:
        return self.vit._get_name()
