import torch
from datasets import Dataset, load_dataset
from torch import Tensor

from xai.datasets.base_dataset import BaseDataset


class CatsDogsCars(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data: Dataset = load_dataset("ENSTA-U2IS/Cats_Dogs_Cars", split="train")  # type: ignore

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
