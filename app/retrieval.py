# retrieval.py
import faiss
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any, Optional


class LegalCaseRetriever:
    def __init__(self, dim: int = 768):
        """
        Initialize a legal case retriever with FAISS vector index.
        
        Args:
            dim: Dimensionality of the embedding vectors (768 for InLegalBERT)
        """
        self.dim = dim
        # Using L2 distance for exact search. For larger indices, consider IVF or HNSW
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # Stores metadata for each vector

    def add_to_index(self, vectors: List[np.ndarray], meta: List[dict]):
        """
        Add embeddings and metadata to the FAISS index.
        
        Args:
            vectors: List of numpy arrays (768,)
            meta: Corresponding metadata dictionaries
        
        Raises:
            ValueError: If vectors and metadata lengths don't match
        """
        if len(vectors) != len(meta):
            raise ValueError("Vectors and metadata lengths do not match.")
        
        # Stack into 2D array and convert to float32 (required by FAISS)
        vectors = np.stack(vectors).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(meta)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, dict]]:
        """
        Perform similarity search in FAISS index.
        
        Args:
            query_vector: Numpy array (768,)
            top_k: Number of top results to return
            
        Returns:
            List of (distance, metadata) tuples
        """
        if not self.index.is_trained or self.index.ntotal == 0:
            return []

        # Ensure correct shape and dtype
        query_vector = query_vector.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Check for valid index
                results.append((dist, self.metadata[idx]))
        return results

    def save_index(self, index_path: str = "faiss_index.index", metadata_path: str = "metadata.json"):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata JSON
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved FAISS index to '{os.path.abspath(index_path)}' and metadata to '{os.path.abspath(metadata_path)}'.")

    def load_index(self, index_path: str = "faiss_index.index", metadata_path: str = "metadata.json"):
        """
        Load the FAISS index and metadata from disk.
        
        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata JSON file
            
        Raises:
            FileNotFoundError: If the index or metadata file doesn't exist
        """
        index_path = os.path.abspath(index_path)
        metadata_path = os.path.abspath(metadata_path)
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded FAISS index from '{index_path}' with {self.index.ntotal} vectors and metadata from '{metadata_path}'.")

    def reset(self):
        """Reset index and metadata."""
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []
        print("Index and metadata reset.")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "vector_count": self.index.ntotal,
            "dimension": self.dim,
            "metadata_count": len(self.metadata),
            "index_type": type(self.index).__name__
        }


# Standalone functions for API usage
def load_faiss_index(index_path: str = "faiss_index.index", metadata_path: str = "metadata.json"):
    """
    Load FAISS index and metadata for use in API.
    
    Args:
        index_path: Path to the FAISS index file
        metadata_path: Path to the metadata JSON file
        
    Returns:
        tuple: (faiss_index, metadata_list)
        
    Raises:
        RuntimeError: If there's an error loading the index
    """
    try:
        # Convert to absolute paths to ensure consistency
        index_path = os.path.abspath(index_path)
        metadata_path = os.path.abspath(metadata_path)
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"One or both files not found: {index_path}, {metadata_path}")
        
        index = faiss.read_index(index_path)
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        print(f"Loaded index from '{index_path}' with {index.ntotal} vectors and {len(metadata)} metadata entries")
        return index, metadata
    except Exception as e:
        raise RuntimeError(f"Error loading FAISS index: {str(e)}")

def search_similar_cases(
    query_vector: np.ndarray, 
    index, 
    metadata: List[Dict[str, Any]], 
    k: int = 3,
    threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar cases using the provided index and metadata.
    
    Args:
        query_vector: The embedded query vector
        index: FAISS index
        metadata: List of metadata dictionaries
        k: Number of results to return
        threshold: Optional similarity threshold (lower is better for L2 distance)
        
    Returns:
        List of metadata dictionaries for the most similar cases
    """
    # Ensure correct shape and data type
    query_vector = query_vector.astype("float32").reshape(1, -1)
    
    # Search the index
    distances, indices = index.search(query_vector, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        # Skip invalid indices
        if idx < 0 or idx >= len(metadata):
            continue
            
        # Apply threshold filtering if provided
        if threshold is not None and dist > threshold:
            continue
            
        # Add distance to metadata for reference
        result = metadata[idx].copy()
        result["distance"] = float(dist)
        results.append(result)
    
    return results
