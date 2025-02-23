import json
from dataclasses import dataclass
from typing import Literal

import yaml

from xai.utils.consts import MODEL_NAMES


@dataclass
class Params:
    # metadata
    model_name: MODEL_NAMES = "alexnet"
    dataset: Literal["cats_dogs_cars"] = "cats_dogs_cars"
    batch_size: int = 64

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        with open(path, "r") as stream:
            params = yaml.safe_load(stream)
            feat_extraction_params = params.get("feature_extraction", {})
            return cls(**feat_extraction_params)

    def to_json(self) -> str:
        filtered_dict = {
            key: val for key, val in self.__dict__.items() if val is not None
        }
        return json.dumps(filtered_dict, indent=2)
