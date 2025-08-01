"""
Configuration management for the FAISS Gemini Embedding system.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()


class Config(BaseModel):
    """Application configuration."""

    # Gemini API settings
    gemini_api_key: str

    # Vector store settings
    persist_path: str = "./data/vector_store"
    embedding_dimension: int = 768

    # Logging settings
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """
        Create configuration from environment variables.

        Returns:
            Config instance

        Raises:
            ValueError: If required environment variables are missing
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Please set it to your Google Gemini API key."
            )

        return cls(
            gemini_api_key=api_key,
            persist_path=os.getenv("PERSIST_PATH", "./data/vector_store"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)


def get_config() -> Config:
    """Get application configuration."""
    return Config.from_env()
