import sys
from pathlib import Path
import pandas as pd
import argparse

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.hybrid.hybrid_recommender import HybridRecommender

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_recommendation(rank, row):
    title = row['names']
    year = int(row['year'])
    score = row['final_score']
    
    sim = row['sim_score_norm']
    qual = row['quality_score_pred']
    clus = row['cluster_score']
    
    print(f"{rank}. {title} ({year})")
    print(f"   Score: {score:.1f}  |  Sim: {sim:.1f}  Qual: {qual:.1f}  Clus: {clus:.1f}")
    print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Hybrid Movie Recommender CLI")
    parser.add_argument("movie", nargs="+", help="Name of the movie you like (e.g., 'John Wick')")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations to show")
    
    args = parser.parse_args()
    movie_title = " ".join(args.movie)
    
    print_header(f"HYBRID MOVIE RECOMMENDER")
    print(f"Searching recommendations for: '{movie_title}'")
    
    try:
        recommender = HybridRecommender()
        
        results = recommender.recommend([movie_title], top_n=args.top_n)
        
        if results.empty:
            print("\nNo results found. Please check if the movie exists in the database.")
        else:
            print_header(f"TOP {len(results)} RECOMMENDATIONS for '{movie_title}'")
            
            for rank, (idx, row) in enumerate(results.iterrows(), 1):
                print_recommendation(rank, row)
                
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    main()
