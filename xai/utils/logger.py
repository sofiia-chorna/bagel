import logging
import os
import time
from pathlib import Path


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = os.path.join(logs_dir, f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ),
    )
    logger.addHandler(file_handler)

    return logger


logger = _configure_logger()
