import os
import textwrap
from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
from matplotlib.figure import Figure
from numpy import ndarray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.logger import logger


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


mean = [0.485, 0.456, 0.406]  # ImageNet mean
std = [0.229, 0.224, 0.225]  # ImageNet std


def get_confusion_matrix(
    model: BaseModel,
    dataloader: DataLoader,
    class_names: Optional[List[str]] = [],
    save_dir: Optional[str] = None,
) -> ndarray[Any, Any]:
    model.to(DEVICE)

    model.eval()
    all_preds = []
    all_labels = []
    misclassified = []

    first_batch_processed = False

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if not first_batch_processed:
                print("gt", labels.cpu().numpy()[0], type(labels.cpu().numpy()[0]))
                print("pred", preds.cpu().numpy()[0], type(preds.cpu().numpy()[0]))

                print("First Batch Details:")
                print(
                    f"Images shape: {images.shape}"
                )  # should be [batch_size, channels, height, width]
                print(f"Ground Truth Labels: {labels.cpu().numpy()}")
                print(f"Predicted Labels: {preds.cpu().numpy()}")

                if class_names:
                    print("Ground Truth Class Names:")
                    print([class_names[label] for label in labels.cpu().numpy()])
                    print("Predicted Class Names:")
                    print([class_names[pred] for pred in preds.cpu().numpy()])

                first_batch_processed = True

            if save_dir is not None:
                # find misclassified samples
                for i in range(len(labels)):
                    if preds[i] != labels[i]:  # prediction is wrong
                        misclassified.append(
                            (images[i].cpu(), labels[i].cpu(), preds[i].cpu())
                        )

    conf_matrix = confusion_matrix(all_labels, all_preds)

    # compute classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    # save metrics to a file
    metrics_path = f"results/{model.get_name()}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    logger.info(f"Saved metrics in {metrics_path}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for idx, (img, true_label, pred_label) in enumerate(misclassified[:100]):
            img = denormalize(img, mean, std)
            img = torch.clamp(img, 0, 1)

            img_pil = transforms.ToPILImage()(img)
            img_pil.save(
                os.path.join(
                    save_dir,
                    f"misclassified_{idx}_true_{class_names[true_label]}_pred_{class_names[pred_label]}.png",
                )
            )
        logger.info(f"Saved {len(misclassified)} misclassified images in '{save_dir}'")

    return conf_matrix


def plot_confusion_matrix(
    conf_matrix: ndarray[Any, Any],
    class_names: Sequence[str],
    title: str = "",
    rotate_labels: bool = True,
    abbreviate_names: bool = False,
    wrap_text: bool = False,
    figsize: tuple[int, int] = (8, 8),
    fontsize: int = 10,
) -> Figure:
    conf_matrix = np.array(conf_matrix, dtype=int)
    plt.figure(figsize=figsize)

    if abbreviate_names:
        class_names = [
            name[:10] + "..." if len(name) > 10 else name for name in class_names
        ]

    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": fontsize},
    )
    ax.collections[0].colorbar.ax.tick_params(labelsize=fontsize)

    if rotate_labels:
        plt.xticks(rotation=45, ha="right", fontsize=fontsize)
        plt.yticks(rotation=0, fontsize=fontsize)

    if wrap_text:
        ax.set_xticklabels(
            ["\n".join(textwrap.wrap(label, 10)) for label in class_names],
            fontsize=fontsize,
        )
        ax.set_yticklabels(
            ["\n".join(textwrap.wrap(label, 10)) for label in class_names],
            fontsize=fontsize,
        )

    plt.title(title, fontsize=fontsize + 4)
    plt.ylabel("actual", fontsize=fontsize)
    plt.xlabel("predicted", fontsize=fontsize)

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    plt.tight_layout()

    return ax.get_figure()  # type: ignore
