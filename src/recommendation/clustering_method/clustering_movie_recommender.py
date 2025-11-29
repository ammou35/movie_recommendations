import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame


@dataclass(frozen=True)
class ClusterDataConfig:
    assignments_path: str
    centroids_path: str
    id_column: str = "id"
    title_column: str = "title"
    cluster_column: str = "cluster_label"
    feature_columns: Sequence[str] = (
        "vote_average",
        "vote_count",
        "release_date",
        "revenue",
        "runtime",
        "adult",
        "budget",
    )


class ClusterWeightedRecommender:
    def __init__(self, config: ClusterDataConfig):
        self.config = config
        self.cluster_assignments = self._load_assignments()
        self.centroids = self._load_centroids()
        self._validate_features()
        self._align_centroids_index()

    def recommend(self, liked_movie_ids: Sequence[int], n_recommendations: int) -> pd.DataFrame:
        liked_movies: DataFrame = self._select_liked_movies(liked_movie_ids)
        if liked_movies.empty or n_recommendations <= 0:
            return self._empty_recommendations()

        cluster_quota: dict = self._compute_cluster_quota(liked_movies, n_recommendations)
        recommendations: DataFrame = self._recommend_from_clusters(liked_movie_ids=liked_movie_ids,
                                                                   cluster_quota=cluster_quota)
        if recommendations.empty:
            return recommendations

        recommendations = (recommendations.sort_values("distance_to_cluster_centroid", ascending=True)
                           .reset_index(drop=True))
        return recommendations

    def _load_assignments(self) -> pd.DataFrame:
        assignments = pd.read_parquet(self.config.assignments_path)
        return assignments

    def _load_centroids(self) -> pd.DataFrame:
        centroids = pd.read_parquet(self.config.centroids_path)
        return centroids

    def _validate_features(self) -> None:
        missing_in_assignments = [
            feature
            for feature in self.config.feature_columns
            if feature not in self.cluster_assignments.columns
        ]
        missing_in_centroids = [
            feature
            for feature in self.config.feature_columns
            if feature not in self.centroids.columns
        ]
        if missing_in_assignments:
            raise ValueError(f"Missing features in assignments: {missing_in_assignments}")
        if missing_in_centroids:
            raise ValueError(f"Missing features in centroids: {missing_in_centroids}")

        required_columns = {
            self.config.id_column,
            self.config.title_column,
            self.config.cluster_column,
        }
        missing_required = [
            column for column in required_columns if column not in self.cluster_assignments.columns
        ]
        if missing_required:
            raise ValueError(f"Missing required columns in assignments: {missing_required}")

    def _align_centroids_index(self) -> None:
        if self.centroids.index.name != self.config.cluster_column:
            self.centroids = self.centroids.copy()
            self.centroids.index.name = self.config.cluster_column

    def _select_liked_movies(self, liked_movie_ids: Sequence[int]) -> pd.DataFrame:
        liked_ids = list(liked_movie_ids)
        mask = self.cluster_assignments[self.config.id_column].isin(liked_ids)
        return self.cluster_assignments.loc[mask]

    def _compute_cluster_quota(
            self,
            liked_movies: pd.DataFrame,
            n_recommendations: int,
    ) -> dict:
        cluster_counts = liked_movies[self.config.cluster_column].value_counts()
        total_liked = int(cluster_counts.sum())
        quota = {}
        for cluster_label, count in cluster_counts.items():
            weight = count / total_liked
            quota[cluster_label] = int(math.ceil(weight * n_recommendations))
        return quota

    def _recommend_from_clusters(self, liked_movie_ids: Sequence[int], cluster_quota: dict, ) -> pd.DataFrame:
        liked_id_set = set(liked_movie_ids)
        all_recommendations = []

        for cluster_label, quota in cluster_quota.items():
            if quota <= 0:
                continue
            cluster_recommendations = self._top_candidates_in_cluster(
                cluster_label=cluster_label,
                quota=quota,
                liked_id_set=liked_id_set,
            )
            if not cluster_recommendations.empty:
                all_recommendations.append(cluster_recommendations)

        if not all_recommendations:
            return self._empty_recommendations()

        return pd.concat(all_recommendations, ignore_index=True)

    def _top_candidates_in_cluster(self, cluster_label: int, quota: int, liked_id_set: set, ) -> pd.DataFrame:
        cluster_movies = self.cluster_assignments[
            self.cluster_assignments[self.config.cluster_column] == cluster_label
            ]
        cluster_movies = cluster_movies[
            ~cluster_movies[self.config.id_column].isin(liked_id_set)
        ]
        if cluster_movies.empty:
            return self._empty_recommendations()

        features = list(self.config.feature_columns)
        feature_matrix = cluster_movies[features].to_numpy(dtype=float)

        try:
            centroid_vector = self.centroids.loc[cluster_label, features].to_numpy(dtype=float)
        except KeyError as error:
            raise KeyError(f"Missing centroid for cluster {cluster_label}") from error

        distances = np.linalg.norm(feature_matrix - centroid_vector, axis=1)
        cluster_movies = cluster_movies.copy()
        cluster_movies["distance_to_cluster_centroid"] = distances

        cluster_movies = cluster_movies.sort_values("distance_to_cluster_centroid", ascending=True)
        top_movies = cluster_movies.head(quota)

        result = top_movies[
            [
                self.config.id_column,
                self.config.title_column,
                self.config.cluster_column,
                "distance_to_cluster_centroid",
            ]
        ].rename(
            columns={
                self.config.id_column: "movie_id",
                self.config.title_column: "title",
            }
        )
        return result.reset_index(drop=True)

    def _empty_recommendations(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["movie_id", "title", self.config.cluster_column, "distance_to_cluster_centroid"])
