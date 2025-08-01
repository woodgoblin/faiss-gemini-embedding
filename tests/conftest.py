"""
Test configuration and fixtures.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.faiss_gemini.vector_store import FaissVectorStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
    ]


@pytest.fixture
def sample_texts():
    """Sample texts corresponding to embeddings."""
    return [
        "This is the first text document",
        "Here is another document about cats",
        "A third document discussing technology",
        "The final document about artificial intelligence",
    ]


@pytest.fixture
def vector_store(temp_dir):
    """Create a vector store instance for testing."""
    return FaissVectorStore(dimension=4, persist_path=temp_dir)


@pytest.fixture
def populated_vector_store(vector_store, sample_embeddings, sample_texts):
    """Vector store populated with sample data."""
    vector_store.add(sample_embeddings, sample_texts)
    return vector_store
