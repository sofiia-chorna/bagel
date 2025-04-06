import torch
import torch.nn as nn

from xai.utils.consts import DEVICE
from xai.utils.logger import logger


def load_from_checkpoint(model: nn.Module, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_state_dict = checkpoint.get("model_state_dict")
    model.load_state_dict(model_state_dict, strict=True)

    logger.info(f"Loaded from checkpoint: {checkpoint_path}")
