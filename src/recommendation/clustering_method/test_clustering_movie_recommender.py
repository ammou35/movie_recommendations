import unittest
from pathlib import Path
import math

import pandas as pd

from src.recommendation.clustering_method.clustering_movie_recommender import (
    ClusterDataConfig,
    ClusterWeightedRecommender,
)


FEATURE_COLUMNS = (
    "vote_average",
    "vote_count",
    "release_date",
    "revenue",
    "runtime",
    "adult",
    "budget",
)


class TestClusterWeightedRecommender(unittest.TestCase):
    def setUp(self) -> None:
        base_path = Path(__file__).resolve().parent
        self.assignments_path = base_path / "assignments.parquet"
        self.centroids_path = base_path / "centroids.parquet"

        config = ClusterDataConfig(
            assignments_path=str(self.assignments_path),
            centroids_path=str(self.centroids_path),
            feature_columns=FEATURE_COLUMNS,
        )
        self.recommender = ClusterWeightedRecommender(config)
        self.assignments = pd.read_parquet(self.assignments_path)

    def test_distributes_recommendations_across_liked_clusters(self) -> None:
        liked_rows = self.assignments.groupby("cluster_label").head(1).head(2)
        liked_ids = liked_rows["id"].tolist()

        n_recommendations = 6
        recommendations = self.recommender.recommend(
            liked_ids, n_recommendations=n_recommendations
        )

        self.assertTrue(
            set(liked_ids).isdisjoint(set(recommendations["movie_id"])),
            "Liked movies must not be recommended back",
        )
        expected_quota = {
            label: int(math.ceil(count / len(liked_ids) * n_recommendations))
            for label, count in liked_rows["cluster_label"].value_counts().items()
        }
        self.assertEqual(
            recommendations["cluster_label"].value_counts().to_dict(),
            expected_quota,
        )
        self.assertTrue(
            recommendations["distance_to_cluster_centroid"].is_monotonic_increasing
        )

    def test_liked_movies_are_excluded(self) -> None:
        liked_ids = self.assignments.groupby("cluster_label").head(3)["id"].tolist()
        n_recommendations = 10
        recommendations = self.recommender.recommend(liked_ids, n_recommendations)

        recommended_ids = set(recommendations["movie_id"].tolist())
        self.assertTrue(set(liked_ids).isdisjoint(recommended_ids))
        self.assertLessEqual(len(recommendations), n_recommendations)
        self.assertFalse(recommendations.empty)

    def test_empty_input_returns_empty_dataframe(self) -> None:
        recommendations = self.recommender.recommend([], n_recommendations=5)
        self.assertTrue(recommendations.empty)
        self.assertListEqual(
            list(recommendations.columns),
            ["movie_id", "title", "cluster_label", "distance_to_cluster_centroid"],
        )

    def test_sample_scenarios_with_prints(self) -> None:
        # Print-friendly scenarios so we can eyeball inputs/outputs of the recommender.
        scenarios = {
            "single_cluster_like": ([266316], 5),
            "two_clusters_even": ([266316, 739864], 6),
            "three_clusters_mix": ([266316, 739864, 4213], 7),
            "multi_like_same_cluster": ([266316, 195877], 5),
        }

        for name, (liked_ids, n_reco) in scenarios.items():
            with self.subTest(name=name):
                print(f"\n--- Scenario: {name} ---")
                print(f"Input liked movie IDs: {liked_ids}")
                recommendations = self.recommender.recommend(
                    liked_ids, n_recommendations=n_reco
                )
                print("Output recommendations:")
                printable = recommendations.to_string(index=False)
                safe_printable = printable.encode("ascii", errors="backslashreplace").decode("ascii")
                print(safe_printable)

                recommended_ids = set(recommendations["movie_id"].tolist())
                liked_clusters = (
                    self.assignments[self.assignments["id"].isin(liked_ids)][
                        "cluster_label"
                    ].nunique()
                )
                self.assertTrue(set(liked_ids).isdisjoint(recommended_ids))
                max_expected = n_reco + liked_clusters - 1  # ceilings can over-allocate by up to k-1
                self.assertLessEqual(len(recommendations), max_expected)
                self.assertFalse(recommendations.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
