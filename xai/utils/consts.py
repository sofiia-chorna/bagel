from typing import Literal

import torch

MODEL_NAMES = Literal["alexnet", "resnet18", "vgg16"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
