from dataclasses import dataclass
from typing import Literal

import yaml


@dataclass
class Params:
    # metadata
    model: Literal["alexnet", "resnet18", "vgg16"] = "alexnet"
    dataset: Literal["cats_dogs_cars"] = "cats_dogs_cars"
    batch_size: int = 64

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        with open(path, "r") as stream:
            params = yaml.safe_load(stream)
            feat_extraction_params = params.get("feature_extraction", {})
            return cls(**feat_extraction_params)
