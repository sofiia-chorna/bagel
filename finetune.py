import os
from dataclasses import asdict

import click
from torch.utils.data import DataLoader, random_split

from xai.datasets.datamodule import HuggingFaceDataModule
from xai.model_operations.finetuning import get_finetuned, plot_training_history
from xai.models.models import get_model
from xai.utils.cli import path
from xai.utils.logger import logger
from xai.utils.params import Params, TrainParams


def split_train_data(
    datamodule: HuggingFaceDataModule,
    batch_size: int = 32,
    val_size: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    train_data = datamodule.train_loader.dataset

    val_len = int(len(train_data) * val_size)  # type: ignore
    train_len = len(train_data) - val_len  # type: ignore

    train_split, val_split = random_split(train_data, [train_len, val_len])

    train_dl = datamodule.get_dataloader(train_split, batch_size)
    val_dl = datamodule.get_dataloader(val_split, batch_size)

    return train_dl, val_dl


@click.group()
def main():
    pass


@main.command()
@path
def finetune(path: str):
    logger.info(f"Running 'fine-tune'")

    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    if params.train_params is None:
        logger.error(
            "To run 'fine-tune' please provide train_params in configuration yaml"
        )
        exit(0)

    datamodule = HuggingFaceDataModule(params.dataset_name, params.batch_size)

    train_loader, val_loader = split_train_data(datamodule, params.batch_size)

    for model_name in params.models:
        model = get_model(model_name, datamodule.num_classes)

        model.prepare_for_finetuning()

        dataset_name = params.dataset_name.split("/")[-1]
        checkpoint_dir = os.path.join("checkpoints", model.get_name(), dataset_name)

        # setup hyperparams
        train_params_dict = asdict(params.train_params)
        model_params = params.get_hyperparams(model_name)
        merged_params_dict = {**train_params_dict, **model_params}
        train_params = TrainParams(**merged_params_dict)

        checkpoint_paths = get_finetuned(
            model,
            train_loader,
            val_loader,
            train_params,
            checkpoint_dir,
        )

        plot_training_history(checkpoint_paths)


if __name__ == "__main__":
    main()
