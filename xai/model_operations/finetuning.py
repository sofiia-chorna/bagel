import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from xai.model_operations.train_monitor import TrainMonitor
from xai.models.base_model import BaseModel
from xai.utils.consts import DEVICE
from xai.utils.logger import logger
from xai.utils.params import TrainParams


def load_checkpoint(
    model: BaseModel,
    checkpoint_path: str,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint.get("val_loss"),
        "val_acc": checkpoint.get("val_acc"),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
    }


def get_finetuned(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    params: TrainParams,
    checkpoint_dir: str = "checkpoints/",
) -> List[str]:
    logger.info(f"Start finetuning {model.get_name()}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    start_epoch = 0
    end_epoch = params.num_epochs

    best_val_loss = float("inf")

    checkpoint_paths = []

    # load weights from checkpoint if available
    if params.checkpoint_path and os.path.exists(params.checkpoint_path):
        checkpoint_dict = load_checkpoint(
            model, params.checkpoint_path, optimizer, scheduler
        )

        best_val_loss = checkpoint_dict.get("best_val_loss", float("inf"))

        start_epoch = checkpoint_dict.get("epoch", 0) + 1
        end_epoch = start_epoch + params.num_epochs

        checkpoint_paths.append(params.checkpoint_path)

        logger.info(f"Resuming training from epoch {start_epoch}")

    model.train()

    monitor = TrainMonitor(patience=5)

    # train
    for epoch in range(start_epoch, end_epoch):
        training_loss = 0.0
        num_batches = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{end_epoch}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            num_batches += 1

        avg_train_loss = training_loss / num_batches

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        logger.info(
            f"Epoch {epoch+1}/{end_epoch}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        should_checkpoint = (
            is_best
            or (epoch + 1) == end_epoch
            or (epoch + 1) % params.save_interval == 0
        )

        if should_checkpoint:
            checkpoint_name = "best.pt" if is_best else f"epoch_{epoch + 1}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_loss": best_val_loss,
                "model_name": model.get_name(),
            }

            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Saved checkpoint to {checkpoint_path}")

            checkpoint_paths.append(checkpoint_path)

            if monitor.should_stop(val_loss):
                break

    return checkpoint_paths


def evaluate(
    model: BaseModel, loader: DataLoader, criterion: nn.CrossEntropyLoss
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += predicted.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total

    return avg_loss, accuracy


def plot_training_history(
    checkpoint_paths: List[str],
    title: str = "training history",
    save_path: str = "train_history.png",
):
    train_losses = []
    val_losses = []
    val_accs = []
    epochs = []

    for path in sorted(checkpoint_paths):
        checkpoint = torch.load(path, map_location="cpu")

        epochs.append(checkpoint["epoch"] + 1)
        train_losses.append(checkpoint["train_loss"])

        val_losses.append(checkpoint["val_loss"])
        val_accs.append(checkpoint["val_acc"])

    plt.figure(figsize=(15, 5))
    plt.title(title)

    plt.plot(epochs, train_losses, label="train loss")
    plt.plot(epochs, val_losses, label="val loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Plot saved to {save_path}")

    plt.close()
