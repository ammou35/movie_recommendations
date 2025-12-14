"""
Semantic Recommender System

Provides movie recommendations based on semantic similarity of descriptions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Union, List, Tuple


class SemanticRecommender:
    """
    Recommend movies based on semantic similarity of descriptions.
    """
    
    def __init__(
        self,
        movies_df: pd.DataFrame,
        embeddings: np.ndarray
    ):
        """
        Initialize the recommender.
        
        Args:
            movies_df: DataFrame with movie information
            embeddings: Numpy array of movie embeddings (n_movies, embedding_dim)
        """
        if len(movies_df) != len(embeddings):
            raise ValueError(
                f"DataFrame length ({len(movies_df)}) must match "
                f"embeddings length ({len(embeddings)})"
            )
        
        self.movies_df = movies_df.reset_index(drop=True)
        self.embeddings = embeddings
        self.similarity_matrix = None
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Precompute similarity matrix for all movies.
        
        Returns:
            Similarity matrix (n_movies, n_movies)
        """
        print("Computing similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        print(f"Similarity matrix computed: shape {self.similarity_matrix.shape}")
        return self.similarity_matrix
    
    def find_movie_index(self, movie_title: str) -> Optional[int]:
        """
        Find the index of a movie by title (case-insensitive partial match).
        
        Args:
            movie_title: Movie title to search for
            
        Returns:
            Index of the movie, or None if not found
        """
        # Try exact match first (case-insensitive)
        exact_match = self.movies_df[
            self.movies_df['names'].str.lower() == movie_title.lower()
        ]
        
        if len(exact_match) > 0:
            return exact_match.index[0]
        
        # Try partial match
        partial_match = self.movies_df[
            self.movies_df['names'].str.lower().str.contains(
                movie_title.lower(), 
                na=False
            )
        ]
        
        if len(partial_match) > 0:
            return partial_match.index[0]
        
        return None
    
    def get_similar_movies(
        self,
        movie_title: str,
        top_n: int = 10,
        return_scores: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Get movies similar to the given movie title.
        
        Args:
            movie_title: Title of the movie to find similar movies for
            top_n: Number of recommendations to return
            return_scores: Whether to return similarity scores
            
        Returns:
            DataFrame with recommended movies (and optionally similarity scores)
        """
        idx = self.find_movie_index(movie_title)
        
        if idx is None:
            partial_matches = self.movies_df[
                self.movies_df['names'].str.lower().str.contains(
                    movie_title.lower(),
                    na=False
                )
            ]
            
            if len(partial_matches) > 0:
                print(f"\nNo exact match found for '{movie_title}'.")
                print("Did you mean one of these?")
                for i, title in enumerate(partial_matches['names'].head(5), 1):
                    print(f"  {i}. {title}")
            else:
                print(f"\nNo movies found matching '{movie_title}'")
            
            return None
        
        movie_name = self.movies_df.loc[idx, 'names']
        print(f"\nFinding recommendations for: {movie_name}")
        
        if self.similarity_matrix is None:
            movie_embedding = self.embeddings[idx].reshape(1, -1)
            similarities = cosine_similarity(movie_embedding, self.embeddings)[0]
        else:
            similarities = self.similarity_matrix[idx]
        
        similar_indices = similarities.argsort()[::-1][1:top_n+1]
        
        recommendations = self.movies_df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = similarities[similar_indices]
        
        # Reorder columns
        display_cols = ['names', 'similarity_score', 'score', 'year', 'genre', 'overview']
        available_cols = [col for col in display_cols if col in recommendations.columns]
        recommendations = recommendations[available_cols]
        
        if return_scores:
            return recommendations
        else:
            return recommendations.drop('similarity_score', axis=1)
    
    def get_recommendations_by_description(
        self,
        description: str,
        embedder,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get movie recommendations based on a text description.
        
        Args:
            description: Text description to find similar movies for
            embedder: SemanticEmbedder instance to encode the description
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        desc_embedding = embedder.encode_single(description)
        
        similarities = cosine_similarity(
            desc_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        similar_indices = similarities.argsort()[::-1][:top_n]
        
        recommendations = self.movies_df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = similarities[similar_indices]
        
        # Reorder columns
        display_cols = ['names', 'similarity_score', 'score', 'year', 'genre', 'overview']
        available_cols = [col for col in display_cols if col in recommendations.columns]
        recommendations = recommendations[available_cols]
        
        return recommendations
    
    def get_batch_recommendations(
        self,
        movie_titles: List[str],
        top_n: int = 10
    ) -> dict:
        """
        Get recommendations for multiple movies at once.
        
        Args:
            movie_titles: List of movie titles
            top_n: Number of recommendations per movie
            
        Returns:
            Dictionary mapping movie titles to their recommendations
        """
        results = {}
        
        for title in movie_titles:
            recommendations = self.get_similar_movies(title, top_n=top_n)
            if recommendations is not None:
                results[title] = recommendations
        
        return results
    
    def get_similarity_score(self, movie_title1: str, movie_title2: str) -> Optional[float]:
        """
        Get similarity score between two movies.
        
        Args:
            movie_title1: First movie title
            movie_title2: Second movie title
            
        Returns:
            Similarity score (0-1), or None if movies not found
        """
        idx1 = self.find_movie_index(movie_title1)
        idx2 = self.find_movie_index(movie_title2)
        
        if idx1 is None or idx2 is None:
            return None
        
        if self.similarity_matrix is not None:
            similarity = self.similarity_matrix[idx1, idx2]
        else:
            emb1 = self.embeddings[idx1].reshape(1, -1)
            emb2 = self.embeddings[idx2].reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0, 0]
        
        return float(similarity)