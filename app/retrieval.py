# retrieval.py
import faiss
import numpy as np
import json
from typing import List, Tuple


class LegalCaseRetriever:
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # L2 distance, you can switch to cosine later if needed
        self.metadata = []  # Stores metadata for each vector

    def add_to_index(self, vectors: List[np.ndarray], meta: List[dict]):
        """
        Add embeddings and metadata to the FAISS index.
        :param vectors: List of numpy arrays (768,)
        :param meta: Corresponding metadata dictionaries
        """
        if len(vectors) != len(meta):
            raise ValueError("Vectors and metadata lengths do not match.")
        
        vectors = np.stack(vectors).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(meta)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, dict]]:
        """
        Perform similarity search in FAISS index.
        :param query_vector: Numpy array (768,)
        :param top_k: Number of top results to return
        :return: List of (distance, metadata)
        """
        if not self.index.is_trained or self.index.ntotal == 0:
            return []

        query_vector = query_vector.astype("float32").reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                results.append((dist, self.metadata[idx]))
        return results

    def save_index(self, index_path: str = "faiss_index.index", metadata_path: str = "metadata.json"):
        """
        Save the FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Saved FAISS index to '{index_path}' and metadata to '{metadata_path}'.")

    def load_index(self, index_path: str = "faiss_index.index", metadata_path: str = "metadata.json"):
        """
        Load the FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"Loaded FAISS index from '{index_path}' and metadata from '{metadata_path}'.")

    def reset(self):
        """Reset index and metadata."""
        self.index.reset()
        self.metadata = []
