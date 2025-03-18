from typing import Dict, List, Optional, Tuple

import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.file import save
from xai.utils.logger import logger


def extract_features(model: BaseModel, loader: DataLoader) -> Dict[str, Tensor]:
    logger.info("Start extracting features")

    model.to(DEVICE)
    model.eval()

    layers: Dict[str, nn.Module] = model.get_layers()

    global_pool = nn.AdaptiveAvgPool2d((1, 1))

    extracted_features: Dict[str, List[Tensor]] = {layer: [] for layer in layers}

    def hook_fn(module: nn.Module, _input: Tuple[Tensor, ...], output: Tensor) -> None:
        pooled_output = global_pool(output).view(output.size(0), -1)
        extracted_features[str(module.name)].append(pooled_output.detach().cpu())

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    for name, layer in layers.items():
        layer.name = name  # type: ignore
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(DEVICE)
            _ = model(images)

    for hook in hooks:
        hook.remove()

    final_features: Dict[str, Tensor] = {
        key: torch.cat(features, dim=0) for key, features in extracted_features.items()
    }

    logger.info(f"End extracting features")

    return final_features


def train_test_split_features(
    concepts_df: DataFrame,
    model_features_dict: Dict[str, Dict[str, Tensor]],
    dataset_name: Optional[str] = "dataset",
):
    train_df, val_df = train_test_split(
        concepts_df, test_size=0.2, random_state=42, stratify=concepts_df["label"]
    )

    train_indices = set(train_df["index"].tolist())
    val_indices = set(val_df["index"].tolist())

    train_features = {}
    val_features = {}

    for model_name, features_dict in model_features_dict.items():
        for layer_name, embeddings in features_dict.items():
            train_features[layer_name] = embeddings[
                [i for i in range(len(embeddings)) if i in train_indices]
            ]
            val_features[layer_name] = embeddings[
                [i for i in range(len(embeddings)) if i in val_indices]
            ]

        save(
            "torch",
            f"features/{dataset_name}/{model_name}_features_train.pt",
            train_features,
        )
        save(
            "torch",
            f"features/{dataset_name}/{model_name}_features_val.pt",
            val_features,
        )

    save("pickle", f"xai/concepts/{dataset_name}_concepts_train.pkl", train_df)
    save("pickle", "xai/concepts/{dataset_name}_concepts_val.pkl", val_df)
