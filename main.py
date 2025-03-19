import click
import pandas as pd
import torch
from datasets import Dataset, load_dataset

from xai.concepts.concept_manager import concept_manager
from xai.concepts.multilabel_classification import (
    run_multilabel_clf,
    run_multilabel_proba,
)
from xai.datasets.datamodule import DataModule
from xai.datasets.imagenet_datamodule import ImageNetDataModule
from xai.evaluation.confusion_matrix import get_confusion_matrix, plot_confusion_matrix
from xai.feature_extraction.feature_extraction import extract_features
from xai.models.models import get_model
from xai.utils.cli import path
from xai.utils.consts import IMAGENET_CLASS_TO_LABEL
from xai.utils.file import load_json, recursive_list_files, save
from xai.utils.logger import logger
from xai.utils.params import Params


@click.group()
def main():
    pass


@main.command()
@path
def annotate(path: str):
    logger.info(f"Running 'annotate'")

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
    logger.info(f"Running 'confusion matrix'")

    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    # dataset
    if params.dataset_type == "imagenet":
        class_filepaths = load_json("imagenet_class_filepaths.json")

        datamodule = ImageNetDataModule(
            params.imagenet_path, class_filepaths, params.batch_size
        )
        datamodule.check_multiple_batches(loader_type="val")
    else:
        datamodule = DataModule(
            params.dataset_type, params.dataset_name, params.batch_size
        )

    # calculate
    for model_name in params.models:
        model = get_model(model_name, datamodule.num_classes)
        _conf_matrix = get_confusion_matrix(
            model, datamodule.val_loader, datamodule.label_names
        )
        """
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
        """


@main.command()
@path
def explain(path: str):
    logger.info(f"Running 'explain'")

    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

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
        if (
            params.train_features_path is not None
            and params.val_features_path is not None
        ):
            logger.info(
                f"Loading features from {params.train_features_path} and {params.val_features_path}"
            )
            train_features = torch.load(params.train_features_path)
            val_features = torch.load(params.val_features_path)
            datamodule = None
        else:
            # dataloaders
            datamodule = DataModule(
                params.dataset_type, params.dataset_name, params.batch_size
            )

            logger.info(f"Calculating features for {model_name}")
            model = get_model(model_name, datamodule.num_classes)

            train_features = extract_features(model, datamodule.train_loader)
            val_features = extract_features(model, datamodule.val_loader)

            base_name = f"{params.dataset_name}_{model_name}"
            save("torch", f"features/{base_name}_train.pt", train_features)
            save("torch", f"features/{base_name}_val.pt", val_features)

        if datamodule is None or datamodule.label_mapping is None:
            label_mapping = IMAGENET_CLASS_TO_LABEL.get
        else:
            label_mapping = datamodule.label_mapping

        #       results = run_multilabel_proba(
        results = run_multilabel_clf(
            train_df=train_concepts_df,
            val_df=val_concepts_df,
            train_features_dict=train_features,
            val_features_dict=val_features,
            label_mapping=label_mapping,
        )

        base_name = f"{params.dataset_name}_{model_name}"
        save("json", f"results/imagenet/{base_name}.json", results)


@main.command()
@path
def run_imagenet_experiment(path: str):
    logger.info(f"Running 'run_imagenet_experiment'")

    params = Params.from_yaml(path)
    logger.info(f"Params used: {params.to_json()}")

    if params.annotations_path is None:
        raise ValueError(
            "Provide 'annotations_path' param to run 'run_imagenet_experiment'"
        )
    if params.imagenet_path is None:
        raise ValueError("Provide 'imagenet_path' to execute 'run_imagenet_experiment'")

    labels_to_keep = params.imagenet_classes

    concepts = []
    class_filepaths = load_json("imagenet_class_filepaths.json")

    if labels_to_keep:
        labels_to_keep = [str(key) for key in labels_to_keep]
        class_filepaths = {
            str(key): value
            for key, value in class_filepaths.items()
            if key in labels_to_keep
        }
        print("class_filepaths", class_filepaths)
        print("labels_to_keep", labels_to_keep)

    index_counter = 0
    for annotation_file_path in recursive_list_files(params.annotations_path):
        logger.info(f"Processing {annotation_file_path} ...")

        data = load_json(annotation_file_path)
        label = IMAGENET_CLASS_TO_LABEL.get(data["imagenet_category_id"], -1)

        if labels_to_keep is not None and str(label) in labels_to_keep:
            for img_entry in data["images"]:
                concepts.append(
                    {
                        "index": index_counter,
                        "label": int(label),
                        "concepts": img_entry["categories"],
                        "filename": img_entry["file_name"],
                    }
                )
                index_counter += 1
        elif labels_to_keep is None:
            for img_entry in data["images"]:
                concepts.append(
                    {
                        "index": index_counter,
                        "label": int(label),
                        "concepts": img_entry["categories"],
                        "filename": img_entry["file_name"],
                    }
                )
                index_counter += 1

    concepts_df = pd.DataFrame(
        concepts, columns=["index", "label", "concepts", "filename"]
    )

    datamodule = ImageNetDataModule(
        params.imagenet_path, class_filepaths, params.batch_size
    )

    for model_name in params.models:
        model = get_model(model_name, datamodule.num_classes)

        train_features = extract_features(model, datamodule.train_loader)
        val_features = extract_features(model, datamodule.val_loader)

        save("torch", f"features/imagenet/{model_name}_train.pt", train_features)
        save("torch", f"features/imagenet/{model_name}_val.pt", val_features)

    train_indices = set(datamodule.train_dataset.indices)
    val_indices = set(datamodule.val_dataset.indices)

    train_df = concepts_df[concepts_df["index"].isin(train_indices)]
    val_df = concepts_df[concepts_df["index"].isin(val_indices)]

    save("pickle", f"concepts/imagenet/concepts_train.pkl", train_df)
    save("pickle", f"concepts/imagenet/concepts_val.pkl", val_df)

    print("concepts_train.pkl", len(train_df))
    print("concepts_vak.pkl", len(val_df))


if __name__ == "__main__":
    main()
