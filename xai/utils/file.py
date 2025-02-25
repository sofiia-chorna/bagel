import json
from pathlib import Path
from typing import Any

from xai.utils.logger import logger


def save_json(file_path: str, value: Any):
    save_path = Path(file_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(value, f, indent=4)

    logger.info(f"Results are saved to {save_path}")
