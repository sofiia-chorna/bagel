import os

import click
import torch

from xai.concepts.concept_manager import Concept_Manager
from xai.concepts.multilabel_classification import run_multilabel_clf
from xai.datasets.datamodule import DataModule
from xai.feature_extraction.feature_extraction import run_extract_features
from xai.models.models import get_model
from xai.utils.cli import path
from xai.utils.file import save_json
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

    extract_feat_params = params.feature_extraction
    if extract_feat_params is None:
        logger.error("No 'feature_extraction' params provided")
        exit(0)

    model = get_model(extract_feat_params.model_name, loader.num_classes)
    run_extract_features(model, loader)

    logger.info("End!")


@main.command()
@path
def compute_multilabel_classification(path: str):
    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    computation_params = params.multilabel_classification
    if computation_params is None:
        logger.error("No 'computation_params' params provided")
        exit(0)

    features_dict = torch.load(computation_params.features_path, weights_only=True)

    concept_manager = Concept_Manager()
    if computation_params.dataframe_path:
        dataframe = concept_manager.load(computation_params.dataframe_path)

    # TODO: implement
    # elif computation_params.dataset:
    # dataframe = concept_manager.extract_concepts(computation_params.dataset)
    # elif computation_params.llm:
    #    dataframe = concept_manager.ask_llm(computation_params.llm)

    else:
        raise ValueError(
            "Please provide either 'dataframe_path', 'dataset' or 'llm' params"
        )

    output_filename = computation_params.output_filename
    results = run_multilabel_clf(dataframe, features_dict)
    save_json(os.path.join("results", f"{output_filename}.json"), results)

    logger.info("End!")


if __name__ == "__main__":
    main()
