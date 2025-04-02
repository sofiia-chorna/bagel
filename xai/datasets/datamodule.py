from torch.utils.data import DataLoader, Dataset

from xai.datasets.hugging_face import HuggingFaceDataset
from xai.utils.logger import logger


class HuggingFaceDataModule:
    train_loader: DataLoader
    val_loader: DataLoader

    def __init__(self, dataset_name: str, batch_size: int, from_disk: bool = False):
        super().__init__()

        logger.info(f"Start creating dataloader for: {dataset_name}")

        train_dataset = HuggingFaceDataset(
            dataset_name, split="train", from_disk=from_disk
        )
        val_dataset = HuggingFaceDataset(
            dataset_name, split="validation", from_disk=from_disk
        )
        self.num_classes = train_dataset.get_num_classes()
        self.label_mapping = train_dataset.get_label_mapping()
        self.label_names = train_dataset.get_label_names()

        self.train_loader = self.get_dataloader(train_dataset, batch_size)
        self.val_loader = self.get_dataloader(val_dataset, batch_size)

        logger.info(f"End creating dataloaders")

    def get_dataloader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
