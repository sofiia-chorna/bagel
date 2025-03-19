from typing import Dict, List, Tuple

import torch
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
