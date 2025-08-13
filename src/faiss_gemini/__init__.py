"""
FAISS-Gemini Embedding System

A system for generating embeddings using Google's Gemini embedding model
and storing/searching them efficiently using FAISS vector database.
"""

from .app import EmbeddingApp
from .config import Config, get_config
from .embedding_service import EmbeddingService
from .mcp_server import EmbeddingMCPServer, run_server
from .vector_store import FaissVectorStore

__version__ = "0.1.0"
__all__ = [
    "EmbeddingService",
    "FaissVectorStore",
    "EmbeddingApp",
    "Config",
    "get_config",
    "EmbeddingMCPServer",
    "run_server",
]
