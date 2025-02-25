from typing import Dict, List, Set

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch import Tensor


def get_multilabel_binarizer(df: DataFrame) -> MultiLabelBinarizer:
    all_concepts: Set[str] = set()

    for concepts in df["concepts"]:
        all_concepts.update(concepts)

    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_concepts)])
    return mlb


def evaluate(
    classifier: MultiOutputClassifier,
    concepts: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    concept_avg_probabilities: Dict[str, float] = {}
    concept_accuracies: Dict[str, float] = {}

    y_pred: np.ndarray = classifier.predict(X_test)  # type: ignore

    for i, concept in enumerate(concepts):
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
    df: DataFrame, features_dict: Dict[str, Tensor]
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    mlb = get_multilabel_binarizer(df)
    results = {}

    for layer, features in features_dict.items():
        df[layer] = list(features.numpy())

        X = np.array(df[layer].to_list())
        y = mlb.transform(df["concepts"].tolist())

        X_train, X_test, y_train, _y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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

        classifier.fit(X_train, y_train)

        layer_results = {}
        for label, group in df.groupby("label"):
            X_test_label = np.array(group[layer].tolist())
            y_test_label = mlb.transform(group["concepts"].tolist())

            layer_results[label] = evaluate(
                classifier=classifier,
                concepts=mlb.classes_,
                X_test=X_test_label,
                y_test=y_test_label,
            )

        results[layer] = layer_results

    return results
