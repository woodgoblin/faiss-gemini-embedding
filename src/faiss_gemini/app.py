"""
Main Application Logic

Combines embedding service and vector store to provide
a complete embedding and search system.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from .embedding_service import EmbeddingResponse, EmbeddingService
from .vector_store import FaissVectorStore, SearchResult

logger = logging.getLogger(__name__)


class EmbeddingApp:
    """Main application for embedding and similarity search."""

    def __init__(
        self,
        api_key: str,
        persist_path: Optional[str] = None,
        embedding_dimension: int = 768,
    ):
        """
        Initialize the embedding application.

        Args:
            api_key: Google Gemini API key
            persist_path: Path to persist the vector store
            embedding_dimension: Dimension of embeddings
        """
        self.embedding_service = EmbeddingService(api_key)
        self.vector_store = FaissVectorStore(
            dimension=embedding_dimension, persist_path=persist_path
        )
        logger.info("Initialized EmbeddingApp")

    async def add_text(self, text: str) -> bool:
        """
        Add a single text to the embedding store.

        Args:
            text: Text to embed and store

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(text)

            # Store in vector database
            self.vector_store.add([embedding], [text])

            logger.info(f"Successfully added text: {text[:100]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to add text: {e}")
            return False

    async def add_texts_batch(self, texts: List[str]) -> int:
        """
        Add multiple texts to the embedding store in batch.

        Args:
            texts: List of texts to embed and store

        Returns:
            Number of texts successfully added
        """
        if not texts:
            return 0

        try:
            # Generate embeddings for all texts
            embedding_responses = (
                await self.embedding_service.generate_embeddings_batch(texts)
            )

            if not embedding_responses:
                logger.warning("No embeddings generated from batch")
                return 0

            # Extract embeddings and texts
            embeddings = [resp.embedding for resp in embedding_responses]
            processed_texts = [resp.text for resp in embedding_responses]

            # Store in vector database
            self.vector_store.add(embeddings, processed_texts)

            logger.info(f"Successfully added {len(embedding_responses)} texts in batch")
            return len(embedding_responses)

        except Exception as e:
            logger.error(f"Failed to add texts batch: {e}")
            return 0

    async def search_similar(self, query_text: str, k: int = 5) -> List[SearchResult]:
        """
        Search for similar texts using cosine similarity.

        Args:
            query_text: Text to search for similar content
            k: Number of results to return

        Returns:
            List of similar texts with scores
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_service.generate_embedding(
                query_text
            )

            # Search in vector store
            results = self.vector_store.search(query_embedding, k)

            logger.info(
                f"Found {len(results)} similar texts for query: {query_text[:100]}..."
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []

    def get_stats(self) -> dict:
        """Get statistics about the application state."""
        stats = self.vector_store.get_stats()
        stats.update(
            {"embedding_model": self.embedding_service.MODEL_NAME, "status": "ready"}
        )
        return stats

    def clear_store(self) -> None:
        """Clear all embeddings from the store."""
        self.vector_store.clear()
        logger.info("Cleared embedding store")

    def save_store(self) -> None:
        """Manually save the store to disk."""
        if self.vector_store.persist_path:
            self.vector_store.save()
        else:
            logger.warning("No persist path configured")

    def load_store(self) -> None:
        """Manually load the store from disk."""
        if self.vector_store.persist_path:
            self.vector_store.load()
        else:
            logger.warning("No persist path configured")
