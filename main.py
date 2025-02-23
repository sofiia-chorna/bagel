import click

from xai.datasets.datamodule import DataModule
from xai.feature_extraction.feature_extraction import run_extract_features
from xai.models.models import get_model
from xai.utils.cli import path
from xai.utils.logger import logger
from xai.utils.params import Params


@click.group()
def main():
    pass


@main.command()
@path
def extract_features(path: str):
    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    loader = DataModule(params)
    model = get_model(params.model_name, loader.num_classes)

    run_extract_features(model, loader)

    logger.info("End!")


if __name__ == "__main__":
    main()
