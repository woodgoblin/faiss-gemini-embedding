"""
Test the Gemini embedding service functionality.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.faiss_gemini.embedding_service import (EmbeddingResponse,
                                                EmbeddingService)


class TestEmbeddingService:
    """Test the EmbeddingService class."""

    @pytest.fixture
    def embedding_service(self):
        """Create an embedding service instance for testing."""
        return EmbeddingService(api_key="test_api_key")

    @pytest.fixture
    def mock_embedding_response(self):
        """Mock response from Gemini API."""
        return {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 150}  # 750 dimensions

    def test_initialization(self):
        """Test embedding service initialization."""
        # Arrange & Act
        service = EmbeddingService(api_key="test_key")

        # Assert
        assert service.api_key == "test_key"
        assert service.MODEL_NAME == "models/embedding-001"

    @pytest.mark.asyncio
    async def test_generate_embedding_success(
        self, embedding_service, mock_embedding_response
    ):
        """Test successful embedding generation."""
        # Arrange
        test_text = "This is a test document"

        with patch(
            "google.generativeai.embed_content", return_value=mock_embedding_response
        ) as mock_embed:
            # Act
            result = await embedding_service.generate_embedding(test_text)

            # Assert
            assert result == mock_embedding_response["embedding"]
            mock_embed.assert_called_once_with(
                model="models/embedding-001",
                content=test_text,
                task_type="retrieval_document",
            )

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, embedding_service):
        """Test error handling for empty text."""
        # Arrange
        empty_texts = ["", "   ", None]

        for empty_text in empty_texts:
            # Act & Assert
            with pytest.raises(ValueError, match="Text cannot be empty"):
                await embedding_service.generate_embedding(empty_text)

    @pytest.mark.asyncio
    async def test_generate_embedding_api_failure(self, embedding_service):
        """Test handling of API failures with retries."""
        # Arrange
        test_text = "Test text"

        with patch(
            "google.generativeai.embed_content", side_effect=Exception("API Error")
        ) as mock_embed:
            # Act & Assert
            with pytest.raises(Exception):
                await embedding_service.generate_embedding(test_text)

            # Should retry 3 times
            assert mock_embed.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_embedding_retry_success(
        self, embedding_service, mock_embedding_response
    ):
        """Test successful embedding generation after retries."""
        # Arrange
        test_text = "Test text"

        # Mock to fail twice, then succeed
        with patch(
            "google.generativeai.embed_content",
            side_effect=[
                Exception("API Error"),
                Exception("API Error"),
                mock_embedding_response,
            ],
        ) as mock_embed:
            # Act
            result = await embedding_service.generate_embedding(test_text)

            # Assert
            assert result == mock_embedding_response["embedding"]
            assert mock_embed.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_success(
        self, embedding_service, mock_embedding_response
    ):
        """Test successful batch embedding generation."""
        # Arrange
        test_texts = ["First text", "Second text", "Third text"]

        with patch(
            "google.generativeai.embed_content", return_value=mock_embedding_response
        ) as mock_embed:
            # Act
            results = await embedding_service.generate_embeddings_batch(test_texts)

            # Assert
            assert len(results) == len(test_texts)
            assert all(isinstance(result, EmbeddingResponse) for result in results)
            assert [result.text for result in results] == test_texts
            assert all(
                result.embedding == mock_embedding_response["embedding"]
                for result in results
            )
            assert mock_embed.call_count == len(test_texts)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_empty(self, embedding_service):
        """Test batch generation with empty list."""
        # Act
        results = await embedding_service.generate_embeddings_batch([])

        # Assert
        assert results == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_partial_failure(
        self, embedding_service, mock_embedding_response
    ):
        """Test batch generation with some failures."""
        # Arrange
        test_texts = ["Success text", "Failure text", "Another success"]

        def mock_embed_side_effect(model, content, task_type):
            if content == "Failure text":
                raise Exception("API Error")
            return mock_embedding_response

        with patch(
            "google.generativeai.embed_content", side_effect=mock_embed_side_effect
        ) as mock_embed:
            # Act
            results = await embedding_service.generate_embeddings_batch(test_texts)

            # Assert
            # Should only return successful embeddings
            assert len(results) == 2  # 2 successful out of 3
            assert results[0].text == "Success text"
            assert results[1].text == "Another success"
            # Each text is called once initially, then the failed one retries 2 more times
            # So: 3 initial calls + 2 retry calls for the failure = 5 total
            assert mock_embed.call_count == 5

    @pytest.mark.asyncio
    async def test_generate_single_embedding_wrapper(
        self, embedding_service, mock_embedding_response
    ):
        """Test the internal single embedding wrapper method."""
        # Arrange
        test_text = "Test text"

        with patch(
            "google.generativeai.embed_content", return_value=mock_embedding_response
        ):
            # Act
            result = await embedding_service._generate_single_embedding(test_text)

            # Assert
            assert isinstance(result, EmbeddingResponse)
            assert result.text == test_text
            assert result.embedding == mock_embedding_response["embedding"]

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(
        self, embedding_service, mock_embedding_response
    ):
        """Test that batch processing is truly concurrent."""
        # Arrange
        test_texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        call_times = []

        def mock_embed_sync(*args, **kwargs):
            import time

            call_times.append(time.time())
            time.sleep(0.01)  # Simulate processing delay
            return mock_embedding_response

        with patch("google.generativeai.embed_content", side_effect=mock_embed_sync):
            # Act
            start_time = asyncio.get_event_loop().time()
            results = await embedding_service.generate_embeddings_batch(test_texts)
            end_time = asyncio.get_event_loop().time()

            # Assert
            assert len(results) == len(test_texts)
            # If truly concurrent, calls should happen roughly at the same time
            # Check that all calls started within a reasonable window
            if len(call_times) > 1:
                time_spread = max(call_times) - min(call_times)
                assert time_spread < 0.05  # All calls should start within 50ms
