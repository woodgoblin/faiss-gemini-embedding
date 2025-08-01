"""
FAISS Vector Store

Handles vector storage and similarity search using FAISS IndexFlatIP
for cosine similarity with persistence support.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Result from similarity search."""

    text: str
    score: float
    index: int


class FaissVectorStore:
    """FAISS-based vector store for embeddings with cosine similarity search."""

    def __init__(self, dimension: int = 768, persist_path: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            dimension: Embedding dimension (default 768 for Gemini)
            persist_path: Path to persist the index and metadata
        """
        self.dimension = dimension
        self.persist_path = persist_path
        self.index = None
        self.texts = []  # Store original texts alongside embeddings

        # Initialize the FAISS index
        self._initialize_index()

        # Load from disk if persistence path exists
        if persist_path and os.path.exists(persist_path):
            self.load()

    def _initialize_index(self):
        """Initialize FAISS IndexFlatIP for cosine similarity."""
        # IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Initialized FAISS IndexFlatIP with dimension {self.dimension}")

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def add(self, embeddings: List[List[float]], texts: List[str]) -> None:
        """
        Add embeddings and their corresponding texts to the store.

        Args:
            embeddings: List of embedding vectors
            texts: List of corresponding texts

        Raises:
            ValueError: If embeddings and texts lengths don't match
        """
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have the same length")

        if not embeddings:
            return

        # Convert to numpy array and normalize for cosine similarity
        vectors = np.array(embeddings, dtype=np.float32)
        normalized_vectors = np.array([self._normalize_vector(vec) for vec in vectors])

        # Add to FAISS index
        self.index.add(normalized_vectors)

        # Store texts
        self.texts.extend(texts)

        logger.info(
            f"Added {len(embeddings)} embeddings to store. Total: {len(self.texts)}"
        )

        # Auto-persist if path is configured
        if self.persist_path:
            self.save()

    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """
        Search for similar embeddings using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of search results sorted by similarity score
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        normalized_query = self._normalize_vector(query_vector[0]).reshape(1, -1)

        # Search using FAISS
        scores, indices = self.index.search(normalized_query, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.texts):  # Valid index
                results.append(
                    SearchResult(
                        text=self.texts[idx], score=float(score), index=int(idx)
                    )
                )

        logger.debug(f"Found {len(results)} similar embeddings")
        return results

    def save(self) -> None:
        """Save the index and metadata to disk."""
        if not self.persist_path:
            raise ValueError("No persist path configured")

        persist_dir = Path(self.persist_path)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = persist_dir / "faiss_index.idx"
        faiss.write_index(self.index, str(index_path))

        # Save metadata (texts)
        metadata_path = persist_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump({"texts": self.texts, "dimension": self.dimension}, f)

        logger.info(f"Saved vector store to {self.persist_path}")

    def load(self) -> None:
        """Load the index and metadata from disk."""
        if not self.persist_path:
            raise ValueError("No persist path configured")

        persist_dir = Path(self.persist_path)
        index_path = persist_dir / "faiss_index.idx"
        metadata_path = persist_dir / "metadata.pkl"

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Index or metadata not found at {self.persist_path}")
            return

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            self.texts = metadata["texts"]
            saved_dimension = metadata["dimension"]

        if saved_dimension != self.dimension:
            logger.warning(
                f"Dimension mismatch: expected {self.dimension}, got {saved_dimension}"
            )

        logger.info(
            f"Loaded vector store from {self.persist_path} with {len(self.texts)} embeddings"
        )

    def clear(self) -> None:
        """Clear all embeddings from the store."""
        self._initialize_index()
        self.texts = []
        logger.info("Cleared vector store")

    def size(self) -> int:
        """Get the number of embeddings in the store."""
        return len(self.texts)

    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            "total_embeddings": len(self.texts),
            "dimension": self.dimension,
            "persist_path": self.persist_path,
            "index_type": type(self.index).__name__,
        }
