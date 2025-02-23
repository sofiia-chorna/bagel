from torch.utils.data import DataLoader, Dataset

from xai.datasets.cats_dogs_cars import CatsDogsCars
from xai.utils.logger import logger
from xai.utils.params import Params


class DataModule:
    loader: DataLoader

    def __init__(self, params: Params):
        super().__init__()

        self.params = params

        logger.info(f"Start creating dataloader for: {self.params.dataset}")
        match self.params.dataset:
            case "cats_dogs_cars":
                dataset = CatsDogsCars()
            case _:
                raise ValueError(f"Unknown dataset: {dataset_name}")

        self.loader = self._get_dataloader(dataset)
        self.num_classes = dataset.get_num_classes()

        logger.info(f"End creating dataloader")

    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.params.batch_size,
            shuffle=shuffle,
        )
