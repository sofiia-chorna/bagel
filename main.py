import click
from datasets import Dataset, load_dataset

from xai.concepts.concept_manager import Concept_Manager
from xai.concepts.multilabel_classification import run_multilabel_clf
from xai.datasets.datamodule import DataModule
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
    concept_manager = Concept_Manager()

    train_concepts_df = concept_manager.ask_llm(train_ds, params.batch_size)
    val_concepts_df = concept_manager.ask_llm(val_ds, params.batch_size)

    # TODO: implement
    # concept_df = concept_manager.extract_concepts(computation_params.dataset)

    logger.info(type(train_concepts_df))
    logger.info(train_concepts_df)

    save("pickle", f"concepts/{params.dataset_name}_train.pkl", train_concepts_df)
    save("pickle", f"concepts/{params.dataset_name}_val.pkl", val_concepts_df)


@main.command()
@path
def explain(path: str):
    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    # dataloaders
    datamodule = DataModule(params.dataset_type, params.dataset_name, params.batch_size)

    # concepts
    concept_manager = Concept_Manager()

    if params.train_concepts_path and params.val_concepts_path:
        train_concepts_df = concept_manager.load(params.train_concepts_path)
        val_concepts_df = concept_manager.load(params.val_concepts_path)
    else:
        logger.error(
            "No 'train_concepts_path' or 'val_concepts_path' provided. Please run annotation first 'python3 main.py annotate'"
        )
        exit(0)

    print(train_concepts_df.head(1)["concepts"])

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
        )
        save("json", f"results/{base_name}.json", results)


if __name__ == "__main__":
    main()
