"""
Gemini Embedding Service

Handles embedding generation using Google's Gemini embedding model
with proper error handling and retry logic.
"""

import asyncio
import logging
from typing import List, Optional

import google.generativeai as genai
from pydantic import BaseModel
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    text: str


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    embedding: List[float]
    text: str


class EmbeddingService:
    """Service for generating embeddings using Gemini embedding model."""

    MODEL_NAME = "models/embedding-001"

    def __init__(self, api_key: str):
        """
        Initialize the embedding service.

        Args:
            api_key: Google Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for given text with retry logic.

        Args:
            text: Input text to embed

        Returns:
            List of embedding values

        Raises:
            Exception: If embedding generation fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        return await self._generate_embedding_with_retry(text)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _generate_embedding_with_retry(self, text: str) -> List[float]:
        """Internal method with retry logic for API calls only."""
        try:
            logger.debug(f"Generating embedding for text: {text[:100]}...")

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model=self.MODEL_NAME, content=text, task_type="retrieval_document"
                ),
            )

            embedding = result["embedding"]
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[EmbeddingResponse]:
        """
        Generate embeddings for multiple texts concurrently.

        Args:
            texts: List of input texts

        Returns:
            List of embedding responses
        """
        if not texts:
            return []

        tasks = [self._generate_single_embedding(text) for text in texts]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to embed text {i}: {result}")
                continue
            embeddings.append(result)

        return embeddings

    async def _generate_single_embedding(self, text: str) -> EmbeddingResponse:
        """Generate single embedding and wrap in response."""
        embedding = await self.generate_embedding(text)
        return EmbeddingResponse(embedding=embedding, text=text)
