"""
Test the main EmbeddingApp functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.faiss_gemini.app import EmbeddingApp
from src.faiss_gemini.vector_store import SearchResult


class TestEmbeddingApp:
    """Test the EmbeddingApp class."""

    @pytest.fixture
    def embedding_app(self, temp_dir):
        """Create an embedding app instance for testing."""
        return EmbeddingApp(
            api_key="test_api_key", persist_path=temp_dir, embedding_dimension=4
        )

    @pytest.fixture
    def mock_embedding(self):
        """Mock embedding vector for testing."""
        return [0.1, 0.2, 0.3, 0.4]

    def test_initialization(self, temp_dir):
        """Test embedding app initialization."""
        # Arrange & Act
        app = EmbeddingApp(
            api_key="test_key", persist_path=temp_dir, embedding_dimension=128
        )

        # Assert
        assert app.embedding_service is not None
        assert app.vector_store is not None
        assert app.vector_store.dimension == 128
        assert app.vector_store.persist_path == temp_dir

    @pytest.mark.asyncio
    async def test_add_text_success(self, embedding_app, mock_embedding):
        """Test successfully adding a single text."""
        # Arrange
        test_text = "This is a test document"

        with patch.object(
            embedding_app.embedding_service,
            "generate_embedding",
            return_value=mock_embedding,
        ) as mock_embed:
            with patch.object(embedding_app.vector_store, "add") as mock_add:
                # Act
                result = await embedding_app.add_text(test_text)

                # Assert
                assert result is True
                mock_embed.assert_called_once_with(test_text)
                mock_add.assert_called_once_with([mock_embedding], [test_text])

    @pytest.mark.asyncio
    async def test_add_text_embedding_failure(self, embedding_app):
        """Test handling embedding generation failure."""
        # Arrange
        test_text = "This is a test document"

        with patch.object(
            embedding_app.embedding_service,
            "generate_embedding",
            side_effect=Exception("API Error"),
        ) as mock_embed:
            # Act
            result = await embedding_app.add_text(test_text)

            # Assert
            assert result is False
            mock_embed.assert_called_once_with(test_text)

    @pytest.mark.asyncio
    async def test_add_text_storage_failure(self, embedding_app, mock_embedding):
        """Test handling vector store failure."""
        # Arrange
        test_text = "This is a test document"

        with patch.object(
            embedding_app.embedding_service,
            "generate_embedding",
            return_value=mock_embedding,
        ) as mock_embed:
            with patch.object(
                embedding_app.vector_store,
                "add",
                side_effect=Exception("Storage Error"),
            ) as mock_add:
                # Act
                result = await embedding_app.add_text(test_text)

                # Assert
                assert result is False
                mock_embed.assert_called_once_with(test_text)
                mock_add.assert_called_once_with([mock_embedding], [test_text])

    @pytest.mark.asyncio
    async def test_add_texts_batch_success(self, embedding_app, mock_embedding):
        """Test successfully adding multiple texts in batch."""
        # Arrange
        test_texts = ["First text", "Second text", "Third text"]
        mock_responses = [
            MagicMock(embedding=mock_embedding, text=text) for text in test_texts
        ]

        with patch.object(
            embedding_app.embedding_service,
            "generate_embeddings_batch",
            return_value=mock_responses,
        ) as mock_batch:
            with patch.object(embedding_app.vector_store, "add") as mock_add:
                # Act
                result = await embedding_app.add_texts_batch(test_texts)

                # Assert
                assert result == len(test_texts)
                mock_batch.assert_called_once_with(test_texts)
                mock_add.assert_called_once_with(
                    [mock_embedding] * len(test_texts), test_texts
                )

    @pytest.mark.asyncio
    async def test_add_texts_batch_empty(self, embedding_app):
        """Test adding empty batch."""
        # Act
        result = await embedding_app.add_texts_batch([])

        # Assert
        assert result == 0

    @pytest.mark.asyncio
    async def test_add_texts_batch_partial_success(self, embedding_app, mock_embedding):
        """Test batch with some successful embeddings."""
        # Arrange
        test_texts = ["First text", "Second text", "Third text"]
        # Only 2 successful embeddings returned
        mock_responses = [
            MagicMock(embedding=mock_embedding, text="First text"),
            MagicMock(embedding=mock_embedding, text="Third text"),
        ]

        with patch.object(
            embedding_app.embedding_service,
            "generate_embeddings_batch",
            return_value=mock_responses,
        ) as mock_batch:
            with patch.object(embedding_app.vector_store, "add") as mock_add:
                # Act
                result = await embedding_app.add_texts_batch(test_texts)

                # Assert
                assert result == 2
                mock_batch.assert_called_once_with(test_texts)
                mock_add.assert_called_once_with(
                    [mock_embedding, mock_embedding], ["First text", "Third text"]
                )

    @pytest.mark.asyncio
    async def test_add_texts_batch_embedding_failure(self, embedding_app):
        """Test handling batch embedding failure."""
        # Arrange
        test_texts = ["First text", "Second text"]

        with patch.object(
            embedding_app.embedding_service,
            "generate_embeddings_batch",
            side_effect=Exception("Batch Error"),
        ) as mock_batch:
            # Act
            result = await embedding_app.add_texts_batch(test_texts)

            # Assert
            assert result == 0
            mock_batch.assert_called_once_with(test_texts)

    @pytest.mark.asyncio
    async def test_search_similar_success(self, embedding_app, mock_embedding):
        """Test successful similarity search."""
        # Arrange
        query_text = "Search query"
        mock_results = [
            SearchResult(text="Similar document 1", score=0.9, index=0),
            SearchResult(text="Similar document 2", score=0.8, index=1),
        ]

        with patch.object(
            embedding_app.embedding_service,
            "generate_embedding",
            return_value=mock_embedding,
        ) as mock_embed:
            with patch.object(
                embedding_app.vector_store, "search", return_value=mock_results
            ) as mock_search:
                # Act
                results = await embedding_app.search_similar(query_text, k=2)

                # Assert
                assert len(results) == 2
                assert results == mock_results
                mock_embed.assert_called_once_with(query_text)
                mock_search.assert_called_once_with(mock_embedding, 2)

    @pytest.mark.asyncio
    async def test_search_similar_embedding_failure(self, embedding_app):
        """Test search with embedding generation failure."""
        # Arrange
        query_text = "Search query"

        with patch.object(
            embedding_app.embedding_service,
            "generate_embedding",
            side_effect=Exception("Embedding Error"),
        ) as mock_embed:
            # Act
            results = await embedding_app.search_similar(query_text)

            # Assert
            assert results == []
            mock_embed.assert_called_once_with(query_text)

    @pytest.mark.asyncio
    async def test_search_similar_storage_failure(self, embedding_app, mock_embedding):
        """Test search with storage failure."""
        # Arrange
        query_text = "Search query"

        with patch.object(
            embedding_app.embedding_service,
            "generate_embedding",
            return_value=mock_embedding,
        ) as mock_embed:
            with patch.object(
                embedding_app.vector_store,
                "search",
                side_effect=Exception("Search Error"),
            ) as mock_search:
                # Act
                results = await embedding_app.search_similar(query_text)

                # Assert
                assert results == []
                mock_embed.assert_called_once_with(query_text)
                mock_search.assert_called_once_with(mock_embedding, 5)

    def test_get_stats(self, embedding_app):
        """Test getting application statistics."""
        # Arrange
        mock_stats = {
            "total_embeddings": 10,
            "dimension": 4,
            "persist_path": "/tmp/test",
            "index_type": "IndexFlatIP",
        }

        with patch.object(
            embedding_app.vector_store, "get_stats", return_value=mock_stats
        ) as mock_get_stats:
            # Act
            stats = embedding_app.get_stats()

            # Assert
            assert "total_embeddings" in stats
            assert "embedding_model" in stats
            assert "status" in stats
            assert stats["embedding_model"] == "models/embedding-001"
            assert stats["status"] == "ready"
            assert stats["total_embeddings"] == 10
            mock_get_stats.assert_called_once()

    def test_clear_store(self, embedding_app):
        """Test clearing the vector store."""
        # Arrange
        with patch.object(embedding_app.vector_store, "clear") as mock_clear:
            # Act
            embedding_app.clear_store()

            # Assert
            mock_clear.assert_called_once()

    def test_save_store_with_path(self, embedding_app):
        """Test manually saving the store."""
        # Arrange
        with patch.object(embedding_app.vector_store, "save") as mock_save:
            # Act
            embedding_app.save_store()

            # Assert
            mock_save.assert_called_once()

    def test_save_store_no_path(self, temp_dir):
        """Test saving when no persist path is configured."""
        # Arrange
        app = EmbeddingApp(api_key="test", persist_path=None)
        app.vector_store.persist_path = None  # Ensure it's None

        with patch.object(app.vector_store, "save") as mock_save:
            # Act
            app.save_store()

            # Assert
            mock_save.assert_not_called()

    def test_load_store_with_path(self, embedding_app):
        """Test manually loading the store."""
        # Arrange
        with patch.object(embedding_app.vector_store, "load") as mock_load:
            # Act
            embedding_app.load_store()

            # Assert
            mock_load.assert_called_once()

    def test_load_store_no_path(self, temp_dir):
        """Test loading when no persist path is configured."""
        # Arrange
        app = EmbeddingApp(api_key="test", persist_path=None)
        app.vector_store.persist_path = None  # Ensure it's None

        with patch.object(app.vector_store, "load") as mock_load:
            # Act
            app.load_store()

            # Assert
            mock_load.assert_not_called()
