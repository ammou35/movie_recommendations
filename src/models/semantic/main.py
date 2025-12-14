"""
MAIN ENTRY POINT - Semantic Movie Recommendation System

This is the main file to run for the semantic recommendation model.
It handles: data loading, preprocessing, embedding generation, and saving.

Usage:
    python src/models/semantic/main.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.movie_data_loader import MovieDataLoader
from models.semantic.text_preprocessor import TextPreprocessor
from models.semantic.embedder import SemanticEmbedder
from models.semantic.recommender import SemanticRecommender
from pathlib import Path
import pandas as pd


def main():
    """
    Complete semantic recommendation pipeline:
    1. Load movie data
    2. Preprocess text descriptions
    3. Generate semantic embeddings
    4. Save embeddings and processed data
    5. Test recommendations
    """
    print("="*80)
    print("SEMANTIC RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
    print("="*80)
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[STEP 1/5] Loading movie data...")
    loader = MovieDataLoader()
    df = loader.get_semantic_data()
    print(f"  Loaded {len(df):,} movies with descriptions")
    
    print("\n[STEP 2/5] Preprocessing text descriptions...")
    preprocessor = TextPreprocessor(min_length=10)
    df_processed = preprocessor.fit_transform(df)
    
    stats = preprocessor.get_stats()
    print(f"  Processed {stats['total_descriptions']:,} descriptions")
    print(f"  Average words per description: {stats['avg_word_count']:.1f}")
    print(f"  Word count range: {stats['min_word_count']:.0f} - {stats['max_word_count']:.0f}")
    
    print("\n[STEP 3/5] Generating semantic embeddings...")
    embedder = SemanticEmbedder(model_name='all-MiniLM-L6-v2')
    embeddings = embedder.generate_embeddings(
        df_processed['overview_processed'],
        batch_size=64,
        show_progress=True
    )
    
    embed_stats = embedder.get_embedding_stats()
    print(f"  Embeddings shape: ({embed_stats['n_samples']}, {embed_stats['embedding_dim']})")
    print(f"  Embedding statistics:")
    print(f"    Mean: {embed_stats['mean']:.4f}")
    print(f"    Std:  {embed_stats['std']:.4f}")
    print(f"    Range: [{embed_stats['min']:.4f}, {embed_stats['max']:.4f}]")
    
    print("\n[STEP 4/5] Saving embeddings and processed data...")
    
    parquet_path = processed_dir / "semantic_embeddings.parquet"
    embedder.save_embeddings(parquet_path, df=df_processed)
    
    npy_path = processed_dir / "semantic_embeddings.npy"
    embedder.save_embeddings(npy_path)
    
    print(f"  Saved to: {parquet_path}")
    print(f"  Saved to: {npy_path}")
    
    print("\n[STEP 5/5] Testing recommendation system...")
    recommender = SemanticRecommender(df_processed, embeddings)
    
    test_movies = [
        "Avatar",
        "Toy Story",
        "The Dark Knight"
    ]
    
    print("\nTesting recommendations for sample movies:")
    for movie in test_movies:
        idx = recommender.find_movie_index(movie)
        if idx is not None:
            movie_title = df_processed.loc[idx, 'names']
            print(f"\n  Movie: {movie_title}")
            
            # Get top 3 recommendations
            recs = recommender.get_similar_movies(movie_title, top_n=3)
            if recs is not None:
                for i, row in recs.iterrows():
                    print(f"    {row['names']} (similarity: {row['similarity_score']:.3f})")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SEMANTIC RECOMMENDATION SYSTEM")
    print("="*80)
    print(f"Total movies processed: {len(df_processed):,}")
    print(f"Embedding dimension: {embed_stats['embedding_dim']}")
    print(f"Data saved to: {parquet_path}")
    print("\nYou can now:")
    print("  1. Use the recommendation system in your code")
    print("  2. Explore results in the Jupyter notebook")
    print("  3. Load embeddings with: pd.read_parquet('data/processed/semantic_embeddings.parquet')")
    print("="*80)


def interactive_mode():
    """
    Interactive mode to test recommendations.
    """
    print("="*80)
    print("SEMANTIC RECOMMENDATION SYSTEM - INTERACTIVE MODE")
    print("="*80)
    
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    parquet_path = project_root / "data" / "processed" / "semantic_embeddings.parquet"
    
    if not parquet_path.exists():
        print("\nError: Embeddings not found. Please run main() first.")
        return
    
    print("\nLoading saved embeddings...")
    df = pd.read_parquet(parquet_path)
    
    embed_cols = [col for col in df.columns if col.startswith('embed_')]
    embeddings = df[embed_cols].values
    
    df_clean = df.drop(columns=embed_cols)
    
    print(f"Loaded {len(df_clean):,} movies")
    
    recommender = SemanticRecommender(df_clean, embeddings)
    
    print("\nEnter a movie title to get recommendations (or 'quit' to exit)")
    
    while True:
        movie_title = input("\nMovie title: ").strip()
        
        if movie_title.lower() in ['quit', 'exit', 'q']:
            break
        
        if not movie_title:
            continue
        
        recommendations = recommender.get_similar_movies(movie_title, top_n=5)
        
        if recommendations is not None:
            print(f"\nTop 5 recommendations:")
            for i, row in recommendations.iterrows():
                print(f"  {row['names']}")
                print(f"    Similarity: {row['similarity_score']:.3f} | "
                      f"Score: {row['score']:.1f} | Year: {int(row['year'])}")
                print(f"    {row['overview'][:100]}...")
                print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        main()