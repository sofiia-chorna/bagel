import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm

from xai.datasets.datamodule import DataModule
from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.logger import logger


def run_extract_features(model: BaseModel, datamodule: DataModule) -> Dict[str, Tensor]:
    logger.info("Start extracting features")

    model.to(DEVICE)
    model.eval()

    layers: Dict[str, nn.Module] = model.get_conv_layers()

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
        for images, _ in tqdm(datamodule.loader):
            images = images.to(DEVICE)
            _ = model(images)

    for hook in hooks:
        hook.remove()

    final_features: Dict[str, Tensor] = {
        key: torch.cat(features, dim=0) for key, features in extracted_features.items()
    }

    save_dir = Path("features")
    save_dir.mkdir(exist_ok=True)
    save_path = os.path.join(save_dir, f"{model.get_name()}_features.pt")
    torch.save(final_features, save_path)

    logger.info(f"Features are saved to {save_path}")

    return final_features
