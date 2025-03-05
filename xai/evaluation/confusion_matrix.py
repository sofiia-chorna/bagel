from typing import Any, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.figure import Figure
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.logger import logger


def get_confusion_matrix(model: BaseModel, dataloader: DataLoader) -> ndarray[Any, Any]:
    logger.info(f"Start evaluating confusion matrix for: {model.get_name()}")

    model.to(DEVICE)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confision matrix: \n {conf_matrix}")

    return conf_matrix


def plot_confusion_matrix(
    conf_matrix: ndarray[Any, Any], class_names: Sequence[str], title: str = ""
) -> Figure:
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("actual")
    plt.xlabel("predicted")

    return ax.get_figure()  # type: ignore
