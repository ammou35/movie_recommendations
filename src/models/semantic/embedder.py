"""
Semantic Embedder for Movie Recommendations

Generates semantic embeddings from movie descriptions using SentenceTransformers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Optional, Union
import pickle


class SemanticEmbedder:
    """
    Generate and manage semantic embeddings for movie descriptions.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = None
    ):
        """
        Initialize the semantic embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embeddings = None
        self.embedding_dim = None
    
    def load_model(self) -> None:
        """Load the SentenceTransformer model."""
        print(f"Loading SentenceTransformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(
        self, 
        descriptions: Union[pd.Series, list],
        batch_size: int = 64,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for movie descriptions.
        
        Args:
            descriptions: Series or list of text descriptions
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        if isinstance(descriptions, pd.Series):
            descriptions = descriptions.tolist()
        
        print(f"Generating embeddings for {len(descriptions)} descriptions...")
        print(f"Batch size: {batch_size}")
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            descriptions,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"Embeddings generated: shape {self.embeddings.shape}")
        
        return self.embeddings
    
    def save_embeddings(
        self, 
        save_path: Union[str, Path],
        df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Save embeddings to disk.
        
        Args:
            save_path: Path to save embeddings (supports .npy, .npz, .parquet)
            df: Optional DataFrame to save alongside embeddings (for .parquet)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.npy':
            np.save(save_path, self.embeddings)
            print(f"Embeddings saved to: {save_path}")
            
        elif save_path.suffix == '.npz':
            np.savez_compressed(save_path, embeddings=self.embeddings)
            print(f"Embeddings saved to: {save_path}")
            
        elif save_path.suffix == '.parquet':
            if df is None:
                raise ValueError("DataFrame required for parquet format")
            
            embedding_cols = [f'embed_{i}' for i in range(self.embeddings.shape[1])]
            embeddings_df = pd.DataFrame(self.embeddings, columns=embedding_cols)
            
            result_df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)
            result_df.to_parquet(save_path, index=False)
            print(f"Embeddings and data saved to: {save_path}")
            
        else:
            raise ValueError(f"Unsupported file format: {save_path.suffix}")
    
    def load_embeddings(self, load_path: Union[str, Path]) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            load_path: Path to load embeddings from
            
        Returns:
            Numpy array of embeddings
        """
        load_path = Path(load_path)
        
        if load_path.suffix == '.npy':
            self.embeddings = np.load(load_path)
            
        elif load_path.suffix == '.npz':
            data = np.load(load_path)
            self.embeddings = data['embeddings']
            
        elif load_path.suffix == '.parquet':
            df = pd.read_parquet(load_path)
            embed_cols = [col for col in df.columns if col.startswith('embed_')]
            self.embeddings = df[embed_cols].values
            
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")
        
        print(f"Embeddings loaded from: {load_path}")
        print(f"Shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def get_embedding_stats(self) -> dict:
        """
        Get statistics about the embeddings.
        
        Returns:
            Dictionary with embedding statistics
        """
        if self.embeddings is None:
            return {}
        
        stats = {
            'n_samples': self.embeddings.shape[0],
            'embedding_dim': self.embeddings.shape[1],
            'mean': float(self.embeddings.mean()),
            'std': float(self.embeddings.std()),
            'min': float(self.embeddings.min()),
            'max': float(self.embeddings.max()),
        }
        
        return stats
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text description.
        
        Args:
            text: Text description to encode
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            self.load_model()
        
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0]