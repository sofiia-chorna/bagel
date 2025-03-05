import click
from datasets import Dataset, load_dataset

from xai.concepts.concept_manager import concept_manager
from xai.concepts.multilabel_classification import run_multilabel_clf
from xai.datasets.datamodule import DataModule
from xai.evaluation.confusion_matrix import (get_confusion_matrix,
                                             plot_confusion_matrix)
from xai.feature_extraction.feature_extraction import extract_features
from xai.models.models import get_model
from xai.utils.cli import path
from xai.utils.file import save
from xai.utils.logger import logger
from xai.utils.params import Params


@click.group()
def main():
    pass


@main.command()
@path
def annotate(path: str):
    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    # dataset
    train_ds: Dataset = load_dataset(params.dataset_name, split="train")  # type: ignore
    val_ds: Dataset = load_dataset(params.dataset_name, split="validation")  # type: ignore

    # concepts
    train_concepts_df = concept_manager.ask_llm(train_ds, params.batch_size)
    val_concepts_df = concept_manager.ask_llm(val_ds, params.batch_size)

    logger.info(type(train_concepts_df))
    logger.info(train_concepts_df)

    save("pickle", f"concepts/{params.dataset_name}_train.pkl", train_concepts_df)
    save("pickle", f"concepts/{params.dataset_name}_val.pkl", val_concepts_df)


@main.command()
@path
def confusion_matrix(path: str):
    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    # dataset
    datamodule = DataModule(params.dataset_type, params.dataset_name, params.batch_size)

    # calculate
    for model_name in params.models:
        model = get_model(model_name, datamodule.num_classes)
        conf_matrix = get_confusion_matrix(model, datamodule.val_loader)
        fig = plot_confusion_matrix(
            conf_matrix,
            datamodule.label_names,
            model_name,
            rotate_labels=True,
            abbreviate_names=True,
            wrap_text=True,
            figsize=(10, 10),
            fontsize=18,
        )

        dataset_name = params.dataset_name.split("/")[1]
        save_path = f"results/confusion_matrix/{dataset_name}_{model_name}.png"
        save("plt", save_path, fig)


@main.command()
@path
def explain(path: str):
    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    # dataloaders
    datamodule = DataModule(params.dataset_type, params.dataset_name, params.batch_size)

    # concepts
    if params.train_concepts_path and params.val_concepts_path:
        train_concepts_df = concept_manager.load(params.train_concepts_path)
        val_concepts_df = concept_manager.load(params.val_concepts_path)
    else:
        logger.error(
            "No 'train_concepts_path' or 'val_concepts_path' provided. Please run annotation first 'python3 main.py annotate'"
        )
        exit(0)

    # calculate
    for model_name in params.models:
        model = get_model(model_name, datamodule.num_classes)

        train_features = extract_features(model, datamodule.train_loader)
        val_features = extract_features(model, datamodule.val_loader)

        base_name = f"{params.dataset_name}_{model_name}"
        save("torch", f"features/{base_name}_train.pt", train_features)
        save("torch", f"features/{base_name}_val.pt", val_features)

        results = run_multilabel_clf(
            train_df=train_concepts_df,
            val_df=val_concepts_df,
            train_features_dict=train_features,
            val_features_dict=val_features,
            label_mapping=datamodule.label_mapping,
        )
        save("json", f"results/{base_name}.json", results)


if __name__ == "__main__":
    main()
