import json
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import yaml


@dataclass
class TrainParams:
    batch_size: int = 64
    num_epochs: int = 5
    lr: float = 0.001
    weight_decay: float = 0.001
    optimizer: str = "adam"
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
    checkpoints: Optional[List[str]] = None
    hyperparams: Optional[dict] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        with open(path, "r") as stream:
            params = yaml.safe_load(stream)
            train_params_data = params.pop("train_params", None)
            hyperparams = params.pop("hyperparams", {})
            train_params = (
                TrainParams(**train_params_data) if train_params_data else None
            )
            return cls(train_params=train_params, hyperparams=hyperparams, **params)

    def get_hyperparams(self, model_name: str):
        return self.hyperparams.get(
            model_name,
            {
                "lr": 0.001,
                "weight_decay": 0.001,
                "batch_size": 64,
                "optimizer": "adam",
            },
        )

    def to_json(self) -> str:
        filtered_dict = {
            key: val for key, val in asdict(self).items() if val is not None
        }
        return json.dumps(filtered_dict, indent=2)
