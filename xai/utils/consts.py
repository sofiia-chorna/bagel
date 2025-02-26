from typing import Literal

import torch

MODEL_NAMES = Literal["alexnet", "resnet18", "vgg16"]
DATASET_TYPES = Literal["hugging_face", "local"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
