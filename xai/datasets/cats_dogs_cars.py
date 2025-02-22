from datasets import load_dataset
from xai.datasets.base_dataset import BaseDataset
import torch
from torch import Tensor


class CatsDogsCars(BaseDataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = load_dataset("ENSTA-U2IS/Cats_Dogs_Cars", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.data[index]
        image = item["image"]
        label = item["label"]

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
