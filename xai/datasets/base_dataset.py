from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torchvision.transforms.v2 as transforms
from torch import Tensor
from torch.utils.data import Dataset

_default_transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class BaseDataset(Dataset, ABC):
    def __init__(self, transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.transform = _default_transform if transform is None else transform

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        pass
