import json
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import yaml


@dataclass
class TrainParams:
    num_epochs: int = 5
    lr: float = 0.001
    weight_decay: float = 0.001
    save_interval: int = 5
    checkpoint_path: Optional[str] = None


@dataclass
class Params:
    # metadata
    batch_size: int = 64
    models: List[str] = field(default_factory=lambda: ["alexnet"])
    dataset_type: Literal["hugging_face"] = "hugging_face"
    dataset_name: str = "imagenet"
    train_features_path: Optional[str] = None
    val_features_path: Optional[str] = None
    train_concepts_path: Optional[str] = None
    val_concepts_path: Optional[str] = None
    annotations_path: Optional[str] = None
    imagenet_path: Optional[str] = None
    imagenet_classes: Optional[List[str]] = None
    train_params: Optional[TrainParams] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        with open(path, "r") as stream:
            params = yaml.safe_load(stream)
            train_params_data = params.pop("train_params", None)
            train_params = (
                TrainParams(**train_params_data) if train_params_data else None
            )
            return cls(train_params=train_params, **params)

    def to_json(self) -> str:
        filtered_dict = {
            key: val for key, val in asdict(self).items() if val is not None
        }
        return json.dumps(filtered_dict, indent=2)
