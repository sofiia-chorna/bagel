import numpy as np
import json
from sklearn.preprocessing import normalize
from itertools import combinations
import seaborn as sns
from xai.utils.logger import logger
from scipy.stats import pearsonr

from typing import List, Dict

import matplotlib.pyplot as plt


def compute_moad_dnn(
    embeddings: np.ndarray,
    concept_annotations: Dict[str, List[str]],
    concepts_c1: List[str],
    concepts_c2: List[str],
    image_ids: List[str],
) -> float:
    """
    Compute MOAD_DNN for two concept groups.

    Parameters:
    - embeddings: np.array of shape (n_images, embedding_dim), L2-normalized
    - concept_annotations: dict mapping image_ids to list of concepts
    - concepts_c1: list of concepts in group C1 (e.g., ['white fur', 'black fur'])
    - concepts_c2: list of concepts in group C2 (e.g., ['snout', 'broad chest'])
    - image_ids: list of image IDs corresponding to embeddings

    Returns:
    - moad: float, MOAD_DNN value
    """
    n_images = embeddings.shape[0]
    if len(image_ids) != n_images:
        raise ValueError(f"Mismatch: {len(image_ids)} IDs vs {n_images} embeddings")

    # Validate annotations
    valid_ids = set(concept_annotations.keys())
    if not all(iid in valid_ids for iid in image_ids):
        logger.warning("Some image IDs lack annotations; skipping invalid IDs")
        valid_indices = [i for i, iid in enumerate(image_ids) if iid in valid_ids]
        embeddings = embeddings[valid_indices]
        image_ids = [image_ids[i] for i in valid_indices]
        n_images = len(valid_indices)

    # Compute pairwise cosine similarities
    similarities = np.dot(embeddings, embeddings.T)

    def compute_sim_diff(concepts):
        """Compute mean similarity difference for a concept group."""
        sim_diff = 0.0
        n_concepts = 0
        for concept in concepts:
            # Images with concept
            X_c = [
                i
                for i, img_id in enumerate(image_ids)
                if concept in concept_annotations.get(img_id, [])
            ]
            # Images without concept
            X_not_c = [i for i in range(n_images) if i not in X_c]

            if len(X_c) < 2 or not X_not_c:
                logger.debug(
                    f"Skipping concept {concept}: insufficient images ({len(X_c)} with, {len(X_not_c)} without)"
                )
                continue

            # Matching pairs (sim_c^+)
            pairs_c = list(combinations(X_c, 2))
            sim_plus = np.mean([similarities[i, j] for i, j in pairs_c])

            # Non-matching pairs (sim_c^-)
            pairs_not_c = [(i, j) for i in X_c for j in X_not_c]
            sim_minus = np.mean([similarities[i, j] for i, j in pairs_not_c])

            sim_diff += sim_plus - sim_minus
            n_concepts += 1

        return sim_diff / n_concepts if n_concepts > 0 else 0.0

    # Compute similarity differences for C1 and C2
    sim_diff_c1 = compute_sim_diff(concepts_c1)
    sim_diff_c2 = compute_sim_diff(concepts_c2)

    # Compute MOAD_DNN
    moad = 0.5 * (sim_diff_c1 - sim_diff_c2)
    return moad


def plot_moad_development(data, save_path="moad_development.png"):
    """
    Plot MOAD_DNN across layers with error bars.
    """
    plt.figure(figsize=(10, 6))
    layers = ["layer0", "layer1", "layer2", "layer3", "layer4"]

    for model_name, model_data in data.items():
        moad_means = [model_data["moad"][layer]["mean"] for layer in layers]
        moad_stds = [model_data["moad"][layer]["std"] for layer in layers]
        plt.errorbar(
            layers,
            moad_means,
            yerr=moad_stds,
            marker="o",
            label=model_name.capitalize(),
            capsize=5,
        )

    plt.xlabel("Layer")
    plt.ylabel("MOAD_DNN")
    plt.title("MOAD_DNN Development Across Layers with Error Bars")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.grid(True, linestyles="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def plot_moad_vs_performance(data, save_path="moad_vs_performance.png"):
    """
    Scatter plot of layer4 MOAD_DNN vs. performance with error bars.
    """
    models = list(data.keys())
    moad_layer4 = [data[model]["moad"]["layer4"]["mean"] for model in models]
    moad_std = [data[model]["moad"]["layer4"]["std"] for model in models]
    performance = [data[model]["performance"]["mean"] for model in models]
    perf_std = [data[model]["performance"]["std"] for model in models]

    corr, p_value = pearsonr(moad_layer4, performance)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=moad_layer4, y=performance, hue=models, style=models, s=100)
    for i, _model in enumerate(models):
        plt.errorbar(
            moad_layer4[i],
            performance[i],
            xerr=moad_std[i],
            yerr=perf_std[i],
            fmt="none",
            capsize=3,
        )

    z = np.polyfit(moad_layer4, performance, 1)
    p = np.poly1d(z)
    plt.plot(
        moad_layer4,
        p(moad_layer4),
        "r--",
        label=f"Regression (r={corr:.2f}, p={p_value:.3f})",
    )

    plt.xlabel("MOAD_DNN (layer4)")
    plt.ylabel("Performance (Accuracy)")
    plt.title("Final Layer MOAD_DNN vs. Model Performance")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
