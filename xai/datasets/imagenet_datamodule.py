from typing import Any, Dict

from torch.utils.data import DataLoader, Dataset

from xai.datasets.imagenet import ImageNetDataset
from xai.utils.logger import logger


class ImageNetDataModule:
    train_loader: DataLoader

    def __init__(
        self,
        src_path: str,
        class_filepaths: Dict[str, Any],
        batch_size: int,
    ):
        super().__init__()

        logger.info(f"Start creating dataloader for imagenet")

        train_dataset = ImageNetDataset(class_filepaths, root=src_path, split="train")
        self.train_loader = self._get_dataloader(train_dataset, batch_size)

        self.num_classes = train_dataset.get_num_classes()

        logger.info(f"End creating dataloaders")

    def _get_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
