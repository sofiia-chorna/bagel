import json
from typing import Literal

import torch

MODEL_NAMES = Literal["alexnet", "resnet18", "vgg16"]
DATASET_TYPES = Literal["hugging_face", "local", "imagenet"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("xai/utils/imagenet_class_index.json", "r") as f:
    imagenet_class_mapping = json.load(f)

IMAGENET_CLASS_TO_LABEL = {
    synset[0]: int(label) for label, synset in imagenet_class_mapping.items()
}

IMAGENET_LABEL_TO_NAME = {
    int(label): synset[1] for label, synset in imagenet_class_mapping.items()
}
