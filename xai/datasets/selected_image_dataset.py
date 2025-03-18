from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class SelectedImageDataset(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        transform: Optional[Callable[[Image], Tensor]] = None,
    ):
        super().__init__()

        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        image_path = self.filepaths[index]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
