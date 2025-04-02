from typing import Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.file import save
from xai.utils.logger import logger


def extract_features(
    model: BaseModel, loader: DataLoader
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
    logger.info("Start extracting features")

    model.to(DEVICE)
    model.eval()

    layers: Dict[str, nn.Module] = model.get_layers()

    global_pool = nn.AdaptiveAvgPool2d((1, 1))

    extracted_features: Dict[str, List[Tensor]] = {layer: [] for layer in layers}

    all_preds = []
    all_labels = []

    def hook_fn(module: nn.Module, _input: Tuple[Tensor, ...], output: Tensor) -> None:
        pooled_output = global_pool(output).view(output.size(0), -1)
        extracted_features[str(module.name)].append(pooled_output.detach().cpu())

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    for name, layer in layers.items():
        layer.name = name  # type: ignore
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    for hook in hooks:
        hook.remove()

    final_features: Dict[str, Tensor] = {
        key: torch.cat(features, dim=0) for key, features in extracted_features.items()
    }

    # compute performance metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    logger.info(
        f"End extracting features with accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1-score: {f1:.4f}"
    )
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    return final_features, metrics
