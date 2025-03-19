from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from xai.datasets.imagenet import ImageNetDataset
from xai.utils.consts import IMAGENET_CLASS_TO_LABEL, IMAGENET_LABEL_TO_NAME
from xai.utils.logger import logger


class ImageNetDataModule:
    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(
        self,
        src_path: str,
        class_filepaths: Dict[str, Any],
        batch_size: int,
    ):
        super().__init__()

        logger.info(f"Start creating dataloaders for ImageNet")

        full_dataset = ImageNetDataset(class_filepaths, root=src_path, split="train")

        dataset_indices = list(range(len(full_dataset)))

        train_indices, val_indices = train_test_split(
            dataset_indices, test_size=0.2, random_state=42
        )

        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        self.train_loader = self._get_dataloader(self.train_dataset, batch_size)
        self.val_loader = self._get_dataloader(self.val_dataset, batch_size)

        self.num_classes = self.get_num_classes()
        self.label_mapping = self.get_label_mapping()
        self.label_names = self.get_label_names()

        logger.info(f"End creating dataloaders")

    def _get_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def get_num_classes(self) -> int:
        """Returns the number of classes in ImageNet"""
        return len(IMAGENET_CLASS_TO_LABEL)

    def get_label_mapping(self) -> Callable[[int], str]:
        """Returns a mapping function from class index to class name"""
        return lambda label: IMAGENET_LABEL_TO_NAME[label]

    def get_label_names(self) -> List[str]:
        """Returns a list of all label names in ImageNet"""
        return [IMAGENET_LABEL_TO_NAME[i] for i in range(self.get_num_classes())]

    def check_multiple_batches(
        self, num_batches: int = 1, loader_type: str = "val"
    ) -> None:
        if loader_type == "train":
            loader = self.train_loader
        elif loader_type == "val":
            loader = self.val_loader
        else:
            raise ValueError("loader_type must be 'train' or 'val'")

        logger.info(f"Checking {num_batches} batches of {loader_type} loader:")

        _fig, axes = plt.subplots(num_batches, 4, figsize=(15, 5 * num_batches))
        axes = axes.flatten()

        # Loop through the specified number of batches
        for batch_idx, (images, labels) in enumerate(loader):
            if batch_idx >= num_batches:
                break

            logger.info(f"Batch {batch_idx + 1}:")
            logger.info(f"Images shape: {images.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(f"Labels: {labels}")

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            denormalized_images = images * std + mean
            denormalized_images = torch.clamp(denormalized_images, 0, 1)

            for i in range(min(4, len(images))):
                ax = axes[batch_idx * 4 + i]
                ax.imshow(denormalized_images[i].permute(1, 2, 0))
                ax.set_title(f"Label: {self.label_mapping(labels[i].item())}")
                ax.axis("off")

        plt.savefig("imagenet_val_multiple_batches.png", dpi=300)
