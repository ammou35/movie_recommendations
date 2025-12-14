"""
Training entry point for the unsupervised movie clusterer.

Loads cleaned clustering features, fits the MovieClusterer, and persists the
fitted instance beside the supervised quality predictor model.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import joblib
from pandas import DataFrame
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.clustering.movie_clusterer import MovieClusterer
from utils.movie_data_loader import load_clustering_data

DEFAULT_NUM_CLUSTERS = 30
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

    save_clusterer(clusterer, model_path)

    silhouette = silhouette_score(data, clusterer.cluster_assignation["cluster_id"])

    print(f"Clusterer trained with fixed k={DEFAULT_NUM_CLUSTERS}.")
    print(f"Silhouette for k={DEFAULT_NUM_CLUSTERS}: {silhouette:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
