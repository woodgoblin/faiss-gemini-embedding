"""
Test the FAISS vector store functionality.
"""

from pathlib import Path

import numpy as np
import pytest

from src.faiss_gemini.vector_store import FaissVectorStore, SearchResult


class TestFaissVectorStore:
    """Test the FaissVectorStore class."""

    def test_initialization(self, temp_dir):
        """Test vector store initialization."""
        # Arrange & Act
        store = FaissVectorStore(dimension=128, persist_path=temp_dir)

        # Assert
        assert store.dimension == 128
        assert store.persist_path == temp_dir
        assert store.index is not None
        assert store.size() == 0
        assert len(store.texts) == 0

    def test_add_single_embedding(self, vector_store):
        """Test adding a single embedding and text."""
        # Arrange
        embedding = [0.1, 0.2, 0.3, 0.4]
        text = "Test document"

        # Act
        vector_store.add([embedding], [text])

        # Assert
        assert vector_store.size() == 1
        assert len(vector_store.texts) == 1
        assert vector_store.texts[0] == text

    def test_add_multiple_embeddings(
        self, vector_store, sample_embeddings, sample_texts
    ):
        """Test adding multiple embeddings."""
        # Arrange & Act
        vector_store.add(sample_embeddings, sample_texts)

        # Assert
        assert vector_store.size() == len(sample_embeddings)
        assert len(vector_store.texts) == len(sample_texts)
        assert vector_store.texts == sample_texts

    def test_add_mismatched_lengths(self, vector_store):
        """Test error handling when embeddings and texts have different lengths."""
        # Arrange
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        texts = ["text1", "text2"]  # Different length

        # Act & Assert
        with pytest.raises(ValueError, match="must have the same length"):
            vector_store.add(embeddings, texts)

    def test_add_empty_lists(self, vector_store):
        """Test adding empty lists."""
        # Arrange & Act
        vector_store.add([], [])

        # Assert
        assert vector_store.size() == 0

    def test_search_similar(self, populated_vector_store, sample_embeddings):
        """Test similarity search functionality."""
        # Arrange
        query_embedding = sample_embeddings[0]  # Use first embedding as query

        # Act
        results = populated_vector_store.search(query_embedding, k=2)

        # Assert
        assert len(results) <= 2
        assert all(isinstance(result, SearchResult) for result in results)
        assert all(hasattr(result, "text") for result in results)
        assert all(hasattr(result, "score") for result in results)
        assert all(hasattr(result, "index") for result in results)

        # First result should be the exact match (highest score)
        if results:
            assert results[0].score >= results[-1].score  # Scores should be descending

    def test_search_empty_store(self, vector_store):
        """Test search on empty store."""
        # Arrange
        query_embedding = [0.1, 0.2, 0.3, 0.4]

        # Act
        results = vector_store.search(query_embedding, k=5)

        # Assert
        assert results == []

    def test_search_more_than_available(self, populated_vector_store):
        """Test searching for more results than available."""
        # Arrange
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        total_embeddings = populated_vector_store.size()

        # Act
        results = populated_vector_store.search(
            query_embedding, k=total_embeddings + 10
        )

        # Assert
        assert len(results) == total_embeddings

    def test_vector_normalization(self, vector_store):
        """Test that vectors are properly normalized for cosine similarity."""
        # Arrange
        embedding = [3.0, 4.0, 0.0, 0.0]  # Length = 5
        text = "Test document"

        # Act
        vector_store.add([embedding], [text])

        # Search with the same vector (should get perfect match)
        results = vector_store.search(embedding, k=1)

        # Assert
        assert len(results) == 1
        # Score should be close to 1.0 (perfect cosine similarity)
        assert abs(results[0].score - 1.0) < 0.001

    def test_clear_store(self, populated_vector_store):
        """Test clearing the vector store."""
        # Arrange
        initial_size = populated_vector_store.size()
        assert initial_size > 0

        # Act
        populated_vector_store.clear()

        # Assert
        assert populated_vector_store.size() == 0
        assert len(populated_vector_store.texts) == 0

    def test_save_and_load(self, temp_dir, sample_embeddings, sample_texts):
        """Test saving and loading persistence."""
        # Arrange
        store1 = FaissVectorStore(dimension=4, persist_path=temp_dir)
        store1.add(sample_embeddings, sample_texts)

        # Act - Save
        store1.save()

        # Create new store and load
        store2 = FaissVectorStore(dimension=4, persist_path=temp_dir)
        store2.load()

        # Assert
        assert store2.size() == len(sample_embeddings)
        assert store2.texts == sample_texts

        # Test search works on loaded store
        query_embedding = sample_embeddings[0]
        results = store2.search(query_embedding, k=1)
        assert len(results) == 1
        assert results[0].text == sample_texts[0]

    def test_auto_persistence(self, temp_dir, sample_embeddings, sample_texts):
        """Test automatic persistence when adding data."""
        # Arrange
        store1 = FaissVectorStore(dimension=4, persist_path=temp_dir)

        # Act
        store1.add(sample_embeddings, sample_texts)

        # Create new store - should auto-load
        store2 = FaissVectorStore(dimension=4, persist_path=temp_dir)

        # Assert
        assert store2.size() == len(sample_embeddings)
        assert store2.texts == sample_texts

    def test_get_stats(self, populated_vector_store):
        """Test getting vector store statistics."""
        # Act
        stats = populated_vector_store.get_stats()

        # Assert
        assert "total_embeddings" in stats
        assert "dimension" in stats
        assert "persist_path" in stats
        assert "index_type" in stats
        assert stats["total_embeddings"] > 0
        assert stats["dimension"] == 4
        assert "IndexFlatIP" in stats["index_type"]

    def test_no_persist_path_operations(self):
        """Test operations when no persist path is configured."""
        # Arrange
        store = FaissVectorStore(dimension=4)  # No persist path

        # Act & Assert
        with pytest.raises(ValueError, match="No persist path configured"):
            store.save()

        with pytest.raises(ValueError, match="No persist path configured"):
            store.load()

    def test_load_nonexistent_files(self, temp_dir):
        """Test loading when files don't exist."""
        # Arrange
        store = FaissVectorStore(
            dimension=4, persist_path=str(Path(temp_dir) / "nonexistent")
        )

        # Act (should not raise exception)
        store.load()

        # Assert
        assert store.size() == 0
