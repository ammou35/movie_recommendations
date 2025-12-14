import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from pandas import DataFrame

# Backward-compatible module aliasing for pickled models that reference
# models.clustering.MovieClusterer (original file name) instead of the current module.
try:
    import models.clustering.movie_clusterer as _mc  # type: ignore

    sys.modules.setdefault("models.clustering.MovieClusterer", _mc)
except Exception:
    pass


class ClusteringMovieRecommender:
    """
    Recommend movies using a pre-trained MovieClusterer saved with joblib.

    The loaded clusterer is expected to expose:
      - cluster_assignation: DataFrame with original features + 'cluster_id'
      - cluster_centroids: DataFrame with columns ['cluster_id', *feature_cols]
    """

    def __init__(self, model_path: str | Path | None = None, model=None):
        if model is not None:
            self.clusterer = model
        elif model_path is not None:
            self.model_path = Path(model_path)
            self.clusterer = joblib.load(self.model_path)
        else:
            raise ValueError("Must provide either model_path or model instance.")

        self.assignations: DataFrame = self.clusterer.cluster_assignation
        self.centroids: DataFrame = self.clusterer.cluster_centroids

        self._feature_cols = [c for c in self.assignations.columns if c != "cluster_id"]
        self._centroid_map: Dict[int, np.ndarray] = {
            int(row["cluster_id"]): row[self._feature_cols].to_numpy(dtype=float)
            for _, row in self.centroids.iterrows()
        }

    def recommend_movies(self, reference_movie_ids: List[int]) -> np.ndarray:
        """
        Score all movies given reference movie IDs.

        Returns
        -------
        np.ndarray
            Array of length number_of_movies with scores in [0, 100].
        """
        number_of_movies = len(self.assignations)
        if not reference_movie_ids:
            return np.zeros(number_of_movies, dtype=float)

        # Filter out IDs not present in the dataset index
        valid_ids = [mid for mid in reference_movie_ids if mid in self.assignations.index]
        if len(valid_ids) == 0:
            return np.zeros(number_of_movies, dtype=float)

        movie_clusters = self.assignations.loc[valid_ids, "cluster_id"]
        cluster_counts = movie_clusters.value_counts()
        cluster_weights = (cluster_counts / cluster_counts.sum()).to_dict()

        # Precompute per-cluster distances to normalize similarity within each cluster
        features = self.assignations[self._feature_cols].to_numpy(dtype=float)
        clusters_all = self.assignations["cluster_id"].to_numpy()
        similarity = np.zeros(number_of_movies, dtype=float)

        for cluster_id, weight in cluster_weights.items():
            mask = clusters_all == cluster_id
            if not mask.any():
                continue

            centroid = self._centroid_map[int(cluster_id)]
            cluster_features = features[mask]
            dists = np.linalg.norm(cluster_features - centroid, axis=1)
            max_dist = dists.max()
            # Similarity in [0,1]; if all points coincide with centroid, similarity is 1
            cluster_similarity = np.ones_like(dists) if max_dist == 0 else 1 - (dists / max_dist)
            similarity[mask] = cluster_similarity * weight

        scores = 100 * similarity

        # Zero out reference movies
        ref_mask = np.zeros(number_of_movies, dtype=bool)
        ref_mask_indices = [self.assignations.index.get_loc(mid) for mid in valid_ids]
        ref_mask[ref_mask_indices] = True
        scores[ref_mask] = 0.0

        return scores
