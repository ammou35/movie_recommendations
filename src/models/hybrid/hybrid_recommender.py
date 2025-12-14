import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.utils.movie_data_loader import MovieDataLoader
from src.models.semantic.recommender import SemanticRecommender
from src.models.clustering.clustering_movie_recommender import ClusteringMovieRecommender
from src.models.supervised.predictor import QualityPredictor
from src.models.supervised.feature_engineering import FeatureEngineer


class HybridRecommender:
    """
    Hybrid Movie Recommender System combining:
    1. Semantic Embedding Model (Content Similarity)
    2. Supervised Quality Model (Quality Prediction)
    3. Clustering Model (Taste Groups/Discovery)
    """
    
    def __init__(self, data_dir: Optional[str] = None, models_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else project_root / "data" / "processed"
        self.models_dir = Path(models_dir) if models_dir else project_root / "src" / "models" / "saved_models"
        
        print("Initializing Hybrid Recommender...")
        self._load_data()
        self._init_semantic_model()
        self._init_quality_model()
        self._init_clustering_model()
        print("Hybrid Recommender Initialized successfully.")

    def _load_data(self):
        """Loads movie data and embeddings."""
        self.loader = MovieDataLoader()
        self.movies_df = self.loader.get_clean_data()
        
        embeddings_path = self.data_dir / "semantic_embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
        self.embeddings = np.load(embeddings_path)

    def _init_semantic_model(self):
        """Initializes the semantic recommender."""
        self.semantic_model = SemanticRecommender(self.movies_df, self.embeddings)

    def _init_quality_model(self):
        """Initializes the supervised quality predictor and feature engineer."""
        supervised_path = self.models_dir / "supervised_model.pkl"
        if not supervised_path.exists():
            raise FileNotFoundError(f"Supervised model not found at {supervised_path}")
            
        self.quality_model = QualityPredictor(str(supervised_path))
        
        model_data = joblib.load(supervised_path)
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names')
        
        self.engineer = FeatureEngineer()
        if self.scaler:
            self.engineer.scaler = self.scaler
            self.engineer.fitted = True

    def _init_clustering_model(self):
        """Initializes the clustering model and prepares name mapping."""
        clustering_path = self.models_dir / "movie_clusterer.pkl"
        self.clustering_model = None
        self._clust_name_to_idx = {}

        if clustering_path.exists():
            try:
                self.clustering_model = ClusteringMovieRecommender(model_path=clustering_path)
                
                if len(self.movies_df) == len(self.clustering_model.assignations):
                    self.clustering_model.assignations['names'] = self.movies_df['names'].values
                
                if "names" in self.clustering_model.assignations.columns:
                    self._clust_name_to_idx = {
                        str(n).lower().strip(): i
                        for i, n in enumerate(self.clustering_model.assignations["names"].values)
                    }
                else:
                    print("WARNING: 'names' column missing in clustering model. Clustering disabled.")
                    self.clustering_model = None
            except Exception as e:
                print(f"WARNING: Failed to load clustering model: {e}")

    def get_semantic_recommendations(self, reference_titles: List[str], top_n: int = 50) -> pd.DataFrame:
        """Get candidates from the semantic model."""
        candidates = pd.DataFrame()
        for title in reference_titles:
            recs = self.semantic_model.get_similar_movies(title, top_n=top_n, return_scores=True)
            if recs is not None:
                candidates = pd.concat([candidates, recs])
        
        if not candidates.empty:
            candidates = candidates.sort_values('similarity_score', ascending=False)
            candidates = candidates.drop_duplicates(subset=['names'])
            candidates['sim_score_norm'] = candidates['similarity_score'] * 100
            
        return candidates

    def get_clustering_ids(self, reference_titles: List[str]) -> List[int]:
        """Convert reference titles to clustering model indices using name mapping."""
        if not self.clustering_model:
            return []
            
        clust_ref_ids = []
        for title in reference_titles:
            clust_idx = self._clust_name_to_idx.get(str(title).lower().strip())
            if clust_idx is not None:
                actual_index_val = self.clustering_model.assignations.index[clust_idx]
                clust_ref_ids.append(actual_index_val)
        return clust_ref_ids

    def get_clustering_recommendations_scores(self, clust_ref_ids: List[int], all_movies: bool = False) -> pd.DataFrame:
        """
        Get cluster scores for provided clustering reference IDs.
        Returns a DataFrame indexed by movie ID with a 'cluster_score' column.
        """
        if not self.clustering_model or not clust_ref_ids:
            return pd.DataFrame()
            
        scores = self.clustering_model.recommend_movies(clust_ref_ids)
        
        return pd.DataFrame({
            'cluster_score': scores
        }, index=self.clustering_model.assignations.index)

    def get_quality_scores(self, candidate_df: pd.DataFrame) -> pd.DataFrame:
        """Predict quality scores for candidates."""
        full_candidates = self.movies_df.loc[candidate_df.index].copy()
        processed_df = self.engineer.create_all_features(full_candidates)
        
        if self.feature_names:
            for col in self.feature_names:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            X = processed_df[self.feature_names]
        else:
            X, _ = self.engineer.prepare_features_and_target(processed_df)
            
        X = X.fillna(0)
        quality_scores = self.quality_model.predict(self.engineer.transform(X))
        
        result_df = candidate_df.copy()
        result_df['quality_score_pred'] = quality_scores
        return result_df

    def recommend(self, reference_titles: List[str], top_n: int = 10, weights: Dict[str, float] = None) -> pd.DataFrame:
        """Main recommendation pipeline."""
        weights = weights or {'similarity': 0.6, 'quality': 0.2, 'cluster': 0.2}
        print(f"\nGeneratring Hybrid Recommendations for: {reference_titles}")
        print("-" * 50)
        
        print(f"1. Getting Semantic Candidates...")
        semantic_recs = self.get_semantic_recommendations(reference_titles, top_n=50)
        print(f"   - Found {len(semantic_recs)} from semantic model")

        print(f"2. Getting Clustering Candidates...")
        clust_ref_ids = self.get_clustering_ids(reference_titles)
        cluster_scores_df = self.get_clustering_recommendations_scores(clust_ref_ids)
        
        cluster_candidates = pd.DataFrame()
        if not cluster_scores_df.empty:
            top_cluster_recs = cluster_scores_df[cluster_scores_df['cluster_score'] > 0.1].sort_values('cluster_score', ascending=False).head(50)
            cluster_candidates = self.movies_df.loc[top_cluster_recs.index].copy()
            cluster_candidates['cluster_score'] = top_cluster_recs['cluster_score']
            print(f"   - Found {len(cluster_candidates)} from clustering model")
        
        ref_indices = []
        for t in reference_titles:
            idx = self.semantic_model.find_movie_index(t)
            if idx is not None: ref_indices.append(idx)
            
        all_indices = pd.Index(semantic_recs.index).union(cluster_candidates.index).difference(ref_indices)
        candidates = self.movies_df.loc[all_indices].copy()
        
        print(f"3. Merging candidates. Total unique pool: {len(candidates)}")
        
        candidates['sim_score_norm'] = 0.0
        candidates['cluster_score'] = 0.0
        candidates['quality_score_pred'] = 0.0
        
        common_sem = candidates.index.intersection(semantic_recs.index)
        candidates.loc[common_sem, 'sim_score_norm'] = semantic_recs.loc[common_sem, 'sim_score_norm']
        
        if not cluster_scores_df.empty:
            clust_names = self.clustering_model.assignations["names"].astype(str).values
            cluster_score_values = cluster_scores_df['cluster_score'].values
            
            score_lookup = {
                str(name).lower().strip(): float(score)
                for name, score in zip(clust_names, cluster_score_values)
            }
            
            cand_names = candidates["names"].astype(str).str.lower().str.strip()
            candidates["cluster_score"] = cand_names.map(score_lookup).fillna(0.0)

        print(f"4. Calculating Quality Scores...")
        candidates = self.get_quality_scores(candidates)
        
        candidates['final_score'] = (
            weights['similarity'] * candidates['sim_score_norm'] +
            weights['quality'] * candidates['quality_score_pred'] +
            weights['cluster'] * candidates['cluster_score']
        )
        
        final_ranking = candidates.sort_values('final_score', ascending=False).head(top_n)
        
        cols = ['names', 'year', 'final_score', 'sim_score_norm', 'quality_score_pred', 'cluster_score']
        return final_ranking[[c for c in cols if c in final_ranking.columns]]
