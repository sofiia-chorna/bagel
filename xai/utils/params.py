import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import yaml

from xai.utils.consts import MODEL_NAMES


@dataclass
class FeatureExtraction:
    model_name: MODEL_NAMES = "alexnet"
    dataset: Literal["cats_dogs_cars"] = "cats_dogs_cars"
    batch_size: int = 64


@dataclass
class MultilabelClassification:
    features_path: str = "features/features.pth"
    dataframe_path: str = "dataframes/dataframe.pth"
    output_filename: str = "results"


@dataclass
class Params:
    feature_extraction: Optional[FeatureExtraction] = None
    multilabel_classification: Optional[MultilabelClassification] = None

    # metadata
    model_name: MODEL_NAMES = "alexnet"
    dataset: Literal["cats_dogs_cars"] = "cats_dogs_cars"
    batch_size: int = 64

    @classmethod
    def from_yaml(cls, path: str) -> "Params":
        with open(path, "r") as stream:
            params = yaml.safe_load(stream)

            feature_extraction = None
            multilabel_classification = None

            # feature extraction params
            feat_extraction_params = params.get("feature_extraction")
            if feat_extraction_params:
                feature_extraction = FeatureExtraction(**feat_extraction_params)

            # multilabel classification params
            multilabel_cls_params = params.get("multilabel_classification")
            if multilabel_cls_params:
                multilabel_classification = MultilabelClassification(
                    **multilabel_cls_params
                )

            return cls(
                feature_extraction=feature_extraction,
                multilabel_classification=multilabel_classification,
            )

    def to_json(self) -> str:
        filtered_dict = {
            key: val for key, val in asdict(self).items() if val is not None
        }
        return json.dumps(filtered_dict, indent=2)
