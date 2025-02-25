from typing import Dict, List, Set

import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch import Tensor


def get_valid_concepts(concepts: List[str], y_concepts: np.ndarray) -> List[int]:
    valid_concepts = []
    for i, _concept in enumerate(concepts):
        if len(np.unique(y_concepts[:, i])) >= 2:
            valid_concepts.append(i)
    return valid_concepts


def get_multilabel_binarizer(df: DataFrame) -> MultiLabelBinarizer:
    all_concepts: Set[str] = set()
    for _, group in df.groupby("label"):
        for concepts in group["concepts"]:
            all_concepts.update(concepts)
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_concepts)])
    return mlb


def evaluate(
    concepts: List[str],
    classifier: MultiOutputClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    concept_avg_probabilities: Dict[str, float] = {}
    concept_accuracies: Dict[str, float] = {}

    y_pred: np.ndarray = classifier.predict(X_test)

    for i, concept in enumerate(concepts):
        # calculate average probability
        binary_clf = classifier.estimators_[i]
        probs: np.ndarray = binary_clf.predict_proba(X_test)
        probs_concept_present: np.ndarray = probs[:, 1]
        concept_avg_probabilities[concept] = probs_concept_present.mean().item()

        # calculate accuracy
        concept_accuracies[concept] = accuracy_score(y_test[:, i], y_pred[:, i])

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

    for layer in features_dict.keys():
        layer_results = {}

        for label, group in df.groupby("label"):
            X = np.array(group[layer].tolist())
            y_concepts = group["concepts"].tolist()

            y_encoded = mlb.transform(y_concepts)

            valid_concepts = get_valid_concepts(mlb.classes_, y_encoded)
            X = X[:, valid_concepts]
            y_encoded = y_encoded[:, valid_concepts]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
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

            layer_results[label] = evaluate(mlb.classes_, classifier, X_test, y_test)

        results[layer] = layer_results

    return results
