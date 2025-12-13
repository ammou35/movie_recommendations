import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.clustering.clustering_movie_recommender import (  # noqa: E402
    ClusteringMovieRecommender,
)
from utils.movie_data_loader import MovieDataLoader  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "movie_clusterer.pkl"


def get_model():
    if not MODEL_PATH.exists():
        pytest.skip("movie_clusterer.pkl not found; run clustering training first.")
    return ClusteringMovieRecommender(MODEL_PATH)


def test_scores_positive_in_reference_cluster_and_zero_elsewhere():
    recommender = get_model()
    assignations: pd.DataFrame = recommender.assignations

    cluster_counts = assignations["cluster_id"].value_counts()
    target_cluster = cluster_counts[cluster_counts >= 2].index[0]
    cluster_ids = assignations[assignations["cluster_id"] == target_cluster].index.tolist()
    reference_id = cluster_ids[0]

    other_cluster_movie = assignations.loc[assignations["cluster_id"] != target_cluster].index[0]

    scores = recommender.recommend_movies([reference_id])

    assert scores.shape == (len(assignations),)
    assert scores.min() >= 0
    assert scores.max() <= 100

    assert scores[assignations.index.get_loc(reference_id)] == 0.0

    same_cluster_mask = assignations["cluster_id"] == target_cluster
    non_ref_mask = same_cluster_mask & (assignations.index != reference_id)
    assert scores[non_ref_mask.to_numpy()].max() > 0

    assert scores[assignations.index.get_loc(other_cluster_movie)] == 0.0


def test_empty_or_unknown_reference_ids_return_zero():
    recommender = get_model()
    assignations: pd.DataFrame = recommender.assignations

    zeros_empty = recommender.recommend_movies([])
    zeros_unknown = recommender.recommend_movies([99999999, -1])

    assert np.allclose(zeros_empty, np.zeros(len(assignations)))
    assert np.allclose(zeros_unknown, np.zeros(len(assignations)))


def test_recommandations_3_different_clusters():
    reference_ids = [0, 1, 2]
    print("\n\nTest avec films ", reference_ids)

    score_and_print_recommendations(reference_ids)


def test_recommandations_vieux_films():
    reference_ids = [87, 58, 22]
    print("\n\nTest avec films ", reference_ids)

    score_and_print_recommendations(reference_ids)


def test_recommandations_all_same_cluster():
    reference_ids = [14, 17, 20]
    print("\n\nTest avec films ", reference_ids)

    score_and_print_recommendations(reference_ids)


def score_and_print_recommendations(reference_ids: list[int]):
    recommender = get_model()
    assignations: pd.DataFrame = recommender.assignations

    # Load clean data to access movie names/metadata aligned by index
    loader = MovieDataLoader()
    clean_df = loader.get_clean_data()

    scores = recommender.recommend_movies(reference_ids)

    # Keep only cluster_id from assignations to avoid column overlap, then append metadata
    columns = ["names", "year", "score", "budget_x", "revenue"]
    results = assignations[["cluster_id"]].join(clean_df[columns], how="left")
    results["recommendation_score"] = scores

    all_cols = ["recommendation_score", "cluster_id"] + columns

    ref_rows = results.loc[reference_ids]
    print("\n--- Reference movies ---")
    print(ref_rows[["cluster_id"] + columns].to_string())

    non_ref = results.loc[~results.index.isin(reference_ids)]
    top10 = non_ref.sort_values("recommendation_score", ascending=False).head(10)

    print("\n--- Top 10 recommendations ---")
    print(top10[all_cols].to_string(index=True))

    assert len(top10) > 0
