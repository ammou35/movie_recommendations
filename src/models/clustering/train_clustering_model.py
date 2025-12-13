"""
Training entry point for the unsupervised movie clusterer.

Loads cleaned clustering features, fits the MovieClusterer, and persists the
fitted instance beside the supervised quality predictor model. Also produces
a 2D PCA scatter plot of the clusters for quick inspection.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Allow imports when running as a script from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.clustering.movie_clusterer import MovieClusterer
from utils.movie_data_loader import load_clustering_data

DEFAULT_NUM_CLUSTERS = 20
DEFAULT_RANDOM_STATE: Optional[int] = 42


def train_clusterer(data: Optional[DataFrame] = None,
                    n_clusters: int = DEFAULT_NUM_CLUSTERS,
                    random_state: Optional[int] = DEFAULT_RANDOM_STATE) -> MovieClusterer:
    """
    Train a MovieClusterer on the cleaned clustering features.
    """
    if data is None:
        data = load_clustering_data()

    clusterer = MovieClusterer(data, number_of_clusters=n_clusters, random_state=random_state)
    clusterer.fit()
    return clusterer


def plot_clusters_2d(
        data: DataFrame,
        cluster_labels,
        n_clusters: int,
        output_path: Path,
        sample_size: int = 5000,
) -> Path:
    """
    Reduce features to 2D with PCA and save a scatter plot colored by cluster.
    """
    plot_df = data.copy()
    plot_df["cluster_id"] = cluster_labels

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=DEFAULT_RANDOM_STATE)

    features = plot_df.drop(columns=["cluster_id"])
    reducer = PCA(n_components=2, random_state=DEFAULT_RANDOM_STATE)
    embedded = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=plot_df["cluster_id"],
        cmap="tab20",
        s=12,
        alpha=0.7,
    )
    ax.set_title(f"Movie clusters (k={n_clusters})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    # Build a legend with one entry per cluster id (instead of a colorbar)
    cmap = plt.get_cmap("tab20")
    handles = []
    labels = []
    for cid in sorted(plot_df["cluster_id"].unique()):
        color = cmap(cid % cmap.N)
        handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="", markersize=6,
                markerfacecolor=color, markeredgecolor=color, alpha=0.7
            )
        )
        labels.append(f"Cluster {cid}")
    ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_clusterer(clusterer: MovieClusterer, output_path: Path) -> Path:
    """
    Persist the fitted clusterer instance to disk with joblib.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clusterer, output_path)
    return output_path


def main():
    data = load_clustering_data()
    clusterer = train_clusterer(data, n_clusters=DEFAULT_NUM_CLUSTERS)

    output_dir = Path(__file__).resolve().parent.parent / "saved_models"
    model_path = output_dir / "movie_clusterer.pkl"
    plot_path = Path(__file__).resolve().parent / "clusters.png"

    save_clusterer(clusterer, model_path)

    plot_clusters_2d(data=data,
                     cluster_labels=clusterer.cluster_assignation["cluster_id"],
                     n_clusters=DEFAULT_NUM_CLUSTERS,
                     output_path=plot_path)

    silhouette = silhouette_score(data, clusterer.cluster_assignation["cluster_id"])

    print(f"Clusterer trained with fixed k={DEFAULT_NUM_CLUSTERS}.")
    print(f"Silhouette for k={DEFAULT_NUM_CLUSTERS}: {silhouette:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
