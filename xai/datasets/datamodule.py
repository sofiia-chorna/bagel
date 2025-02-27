from torch.utils.data import DataLoader, Dataset

from xai.datasets.hugging_face import HuggingFaceDataset
from xai.utils.consts import DATASET_TYPES
from xai.utils.logger import logger


class DataModule:
    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(self, dataset_type: DATASET_TYPES, dataset_name: str, batch_size: int):
        super().__init__()

        logger.info(f"Start creating dataloader for: {dataset_name}")

        if dataset_type in ["hugging_face", "local"]:
            from_disk = dataset_type == "local"

            train_dataset = HuggingFaceDataset(
                dataset_name, split="train", from_disk=from_disk
            )
            val_dataset = HuggingFaceDataset(
                dataset_name, split="validation", from_disk=from_disk
            )
            self.num_classes = train_dataset.get_num_classes()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        self.train_loader = self._get_dataloader(train_dataset, batch_size)
        self.val_loader = self._get_dataloader(val_dataset, batch_size)

        logger.info(f"End creating dataloaders")

    def _get_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
