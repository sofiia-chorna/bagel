import json
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import yaml


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

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        with open(path, "r") as stream:
            params = yaml.safe_load(stream)
            return cls(**params)

    def to_json(self) -> str:
        filtered_dict = {
            key: val for key, val in asdict(self).items() if val is not None
        }
        return json.dumps(filtered_dict, indent=2)
