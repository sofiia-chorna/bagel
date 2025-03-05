from typing import Callable, List

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch import Tensor

from xai.datasets.base_dataset import BaseDataset


class HuggingFaceDataset(BaseDataset):
    def __init__(
        self, name: str, split: str = "train", from_disk: bool = False
    ) -> None:
        super().__init__()

        if from_disk:
            dataset: DatasetDict = load_from_disk(name)  # type: ignore
            self.data: Dataset = dataset[split]
        else:
            self.data: Dataset = load_dataset(name, split=split)  # type: ignore

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.data[index]
        image = item["image"]
        label = item["label"]

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self) -> int:
        return len(self.data.unique("label"))

    def get_label_mapping(self) -> Callable[[int], str]:
        return self.data.features["label"].int2str

    def get_label_names(self) -> List[str]:
        return self.data.features["label"].names
