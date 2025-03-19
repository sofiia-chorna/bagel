from typing import Callable, Dict, List, Set

import joblib
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch import Tensor

from xai.concepts.concept_manager import concept_manager
from xai.utils.file import save
from xai.utils.logger import logger
from xai.utils.consts import IMAGENET_CLASS_TO_LABEL



def get_multilabel_binarizer(df: DataFrame) -> MultiLabelBinarizer:
    all_concepts: Set[str] = set()

    for concepts in df["concepts"]:
        all_concepts.update(concepts)

    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_concepts)])
    logger.info(f"Fitted MultiLabelBinarizer with {len(all_concepts)} classes")
    return mlb


def evaluate(
    classifier: MultiOutputClassifier,
    concepts: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    logger.info("Evaluating classifier")
    concept_avg_probabilities: Dict[str, float] = {}
    concept_accuracies: Dict[str, float] = {}

    y_pred: np.ndarray = classifier.predict(X_test)  # type: ignore

    for i, concept in enumerate(concepts):
        logger.info(f"start predicting proba for {concept} ({i} / {len(concepts)})")
        # calculate average probability
        binary_clf: BaseEstimator = classifier.estimators_[i]
        probs: np.ndarray = binary_clf.predict_proba(X_test)  # type: ignore
        concept_avg_probabilities[concept] = probs[:, 1].mean().item()

        # calculate accuracy
        concept_accuracies[concept] = accuracy_score(y_test[:, i], y_pred[:, i])  # type: ignore

    return {
        "probability": concept_avg_probabilities,
        "concept_accuracies": concept_accuracies,
    }


def run_multilabel_clf(
    train_df: DataFrame,
    val_df: DataFrame,
    train_features_dict: Dict[str, Tensor],
    val_features_dict: Dict[str, Tensor],
    label_mapping: Callable[[int], str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]:
    logger.info("Starting multilabel classification")

    mlb = get_multilabel_binarizer(train_df)
    results = {}

    for layer, train_features in train_features_dict.items():
        if layer in ["layer4"]:
            logger.info(f"Processing layer {layer}")

            train_df[layer] = list(train_features.numpy())
            val_df[layer] = list(val_features_dict[layer].numpy())

            logger.info("list numpy convert")

            X_train = np.array(train_df[layer].to_list())
            y_train = mlb.transform(train_df["concepts"].tolist())

            logger.info("Start creating StandardScaler")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            logger.info("Scaler is fitted and data is transformed")

            classifier = MultiOutputClassifier(
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear", class_weight="balanced", max_iter=1000, C=0.1
                    )
                ),
                n_jobs=-1
            )
            classifier.fit(X_train_scaled, y_train)
            logger.info("Classifier is trained")

            joblib.dump(classifier, f"classifier_checkpoint_layer_{layer}.pkl")
            joblib.dump(scaler, f"scaler_checkpoint_layer_{layer}.pkl")

            layer_results = {}
            for label, group in val_df.groupby("label", sort=False):
                X_test_label = np.array(group[layer].tolist())

                X_test_label = scaler.transform(X_test_label)

                y_test_label = mlb.transform(group["concepts"].tolist())

                label_results = evaluate(
                    classifier=classifier,
                    concepts=mlb.classes_,
                    X_test=X_test_label,
                    y_test=y_test_label,
                )

                save("json", f"results/imagenet/resnet18_{layer}_{label}.json", label_results)

                label_name = IMAGENET_CLASS_TO_LABEL.get(int(label))
                layer_results[label_name] = format_by_category(label_results)

            results[layer] = layer_results
            save("json", f"results/imagenet/resnet18_{layer}.json", layer_results)

        logger.info("Completed multilabel classification")

    return results


def run_multilabel_proba(
    train_df: DataFrame,
    val_df: DataFrame,
    train_features_dict: Dict[str, Tensor],
    val_features_dict: Dict[str, Tensor],
    label_mapping: Callable[[int], str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]:
    logger.info("Start multilabel classification")

    mlb = get_multilabel_binarizer(train_df)
    results = {}

    last_layer = list(train_features_dict.items())[-1]
    layer, train_features = last_layer

    train_df[layer] = list(train_features.numpy())
    val_df[layer] = list(val_features_dict[layer].numpy())

    X_train = np.array(train_df[layer].to_list())
    y_train = mlb.transform(train_df["concepts"].tolist())
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    classifier = MultiOutputClassifier(
        OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                C=0.1,
                n_jobs=-1
            )
        )
    )
    classifier.fit(X_train, y_train)

    layer_results = {}
    for label, group in val_df.groupby("label", sort=False):
        X_test_label = np.array(group[layer].tolist())
        X_test_label = scaler.transform(X_test_label)
        y_test_label = mlb.transform(group["concepts"].tolist())

        concept_avg_probabilities: Dict[str, np.ndarray] = {}

        # calculate probability
        for i, concept in enumerate(mlb.classes_):
                binary_clf: BaseEstimator = classifier.estimators_[i]
                probs: np.ndarray = binary_clf.predict_proba(X_test_label)  # type: ignore
                concept_avg_probabilities[concept] = probs[:, 1].tolist()

                label_name = label_mapping(int(label))
                layer_results[label_name] = {
                    "probability": concept_avg_probabilities,
                }

    results[layer] = layer_results

    logger.info("End multilabel classification")

    return results


def format_by_category(label_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    category_results = {
        "probability": {},
        "concept_accuracies": {},
    }

    for category, concepts_in_category in concept_manager.all_concepts.items():
        category_results["probability"][category] = {
            concept: label_results["probability"][concept]
            for concept in concepts_in_category
            if concept in label_results["probability"]
        }

        category_results["concept_accuracies"][category] = {
            concept: label_results["concept_accuracies"][concept]
            for concept in concepts_in_category
            if concept in label_results["concept_accuracies"]
        }

    return category_results


def run_multilabel_clf_by_class(
    train_df: DataFrame,
    val_df: DataFrame,
    train_features_dict: Dict[str, Tensor],
    val_features_dict: Dict[str, Tensor],
    label_mapping: Callable[[int], str],
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    logger.info("Start multilabel classification")

    mlb = get_multilabel_binarizer(train_df)
    results = {}

    for layer, train_features in train_features_dict.items():
        train_df[layer] = list(train_features.numpy())
        val_df[layer] = list(val_features_dict[layer].numpy())

        layer_results = {}

        # Train per category (label)
        for label, group in train_df.groupby("label", sort=False):
            X_train_label = np.array(group[layer].tolist())
            y_train_label = mlb.transform(group["concepts"].tolist())

            scaler = StandardScaler()
            X_train_label = scaler.fit_transform(X_train_label)

            classifier = MultiOutputClassifier(
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        C=0.1,
                    )
                )
            )
            classifier.fit(X_train_label, y_train_label)

            # Evaluate per category
            X_test_label = np.array(val_df[val_df["label"] == label][layer].tolist())
            y_test_label = mlb.transform(val_df[val_df["label"] == label]["concepts"].tolist())

            X_test_label = scaler.transform(X_test_label)

            label_results = evaluate(
                classifier=classifier,
                concepts=mlb.classes_,
                X_test=X_test_label,
                y_test=y_test_label,
            )
    
            label_name = label_mapping(int(label))
            layer_results[label_name] = format_by_category(label_results)

        results[layer] = layer_results

    logger.info("End multilabel classification")

    return results
