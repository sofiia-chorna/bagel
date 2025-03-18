import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Literal

import torch

from xai.utils.logger import logger


def save(type: Literal["json", "torch", "pickle", "plt"], file_path: str, value: Any):
    save_path = Path(file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    match type:
        case "json":
            with open(save_path, "w") as f:
                json.dump(value, f, indent=4)

        case "torch":
            torch.save(value, save_path)

        case "pickle":
            with open(save_path, "wb") as f:
                pickle.dump(value, f)

        case "plt":
            value.savefig(save_path, dpi=300)

    logger.info(f"Results are saved to {save_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)


def recursive_list_files(path: str):
    """
    Recursively yield the file paths in a directory.
    """
    for root, _, files in os.walk(path):
        for file in files:
            yield os.path.join(root, file)
