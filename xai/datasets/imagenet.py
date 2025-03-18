import os
from typing import Tuple

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder

from xai.datasets.base_dataset import BaseDataset
from xai.datasets.selected_image_dataset import SelectedImageDataset


class ImageNetDataset(BaseDataset):
    def __init__(
        self,
        class_filepaths: dict,  # {class_name: [file_paths]}
        root: str = "./ImageNet/",
        split: str = "train",
    ) -> None:
        super().__init__()

        self.root = os.path.join(root, split)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.filepaths = []
        self.labels = []

        for label, paths in class_filepaths.items():
            self.filepaths.extend([os.path.join(self.root, path) for path in paths])
            self.labels.extend([label] * len(paths))

        self.data = SelectedImageDataset(
            self.filepaths, self.labels, transform=self.transform
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image, label = self.data[index]
        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self) -> int:
        return len(self.data.classes) if isinstance(self.data, ImageFolder) else 1
