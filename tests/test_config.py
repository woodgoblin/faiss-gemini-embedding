"""
Test the configuration management functionality.
"""

import os
from unittest.mock import patch

import pytest

from src.faiss_gemini.config import Config, get_config


class TestConfig:
    """Test the Config class."""

    def test_config_initialization(self):
        """Test direct config initialization."""
        # Arrange & Act
        config = Config(
            gemini_api_key="test_key",
            persist_path="/test/path",
            embedding_dimension=256,
            log_level="DEBUG",
        )

        # Assert
        assert config.gemini_api_key == "test_key"
        assert config.persist_path == "/test/path"
        assert config.embedding_dimension == 256
        assert config.log_level == "DEBUG"

    def test_config_defaults(self):
        """Test config with default values."""
        # Arrange & Act
        config = Config(gemini_api_key="test_key")

        # Assert
        assert config.gemini_api_key == "test_key"
        assert config.persist_path == "./data/vector_store"
        assert config.embedding_dimension == 768
        assert config.log_level == "INFO"

    def test_from_env_success(self):
        """Test creating config from environment variables."""
        # Arrange
        env_vars = {
            "GEMINI_API_KEY": "env_test_key",
            "PERSIST_PATH": "/env/test/path",
            "EMBEDDING_DIMENSION": "512",
            "LOG_LEVEL": "WARNING",
        }

        with patch.dict(os.environ, env_vars):
            # Act
            config = Config.from_env()

            # Assert
            assert config.gemini_api_key == "env_test_key"
            assert config.persist_path == "/env/test/path"
            assert config.embedding_dimension == 512
            assert config.log_level == "WARNING"

    def test_from_env_minimal(self):
        """Test creating config with only required environment variables."""
        # Arrange
        env_vars = {"GEMINI_API_KEY": "minimal_key"}

        with patch.dict(os.environ, env_vars, clear=True):
            # Act
            config = Config.from_env()

            # Assert
            assert config.gemini_api_key == "minimal_key"
            assert config.persist_path == "./data/vector_store"  # Default
            assert config.embedding_dimension == 768  # Default
            assert config.log_level == "INFO"  # Default

    def test_from_env_missing_api_key(self):
        """Test error when API key is missing."""
        # Arrange
        with patch.dict(os.environ, {}, clear=True):
            # Act & Assert
            with pytest.raises(
                ValueError, match="GEMINI_API_KEY environment variable is required"
            ):
                Config.from_env()

    def test_from_env_empty_api_key(self):
        """Test error when API key is empty."""
        # Arrange
        env_vars = {"GEMINI_API_KEY": ""}

        with patch.dict(os.environ, env_vars, clear=True):
            # Act & Assert
            with pytest.raises(
                ValueError, match="GEMINI_API_KEY environment variable is required"
            ):
                Config.from_env()

    def test_from_env_invalid_dimension(self):
        """Test handling of invalid embedding dimension."""
        # Arrange
        env_vars = {"GEMINI_API_KEY": "test_key", "EMBEDDING_DIMENSION": "not_a_number"}

        with patch.dict(os.environ, env_vars):
            # Act & Assert
            with pytest.raises(ValueError):
                Config.from_env()

    def test_from_dict(self):
        """Test creating config from dictionary."""
        # Arrange
        config_dict = {
            "gemini_api_key": "dict_key",
            "persist_path": "/dict/path",
            "embedding_dimension": 1024,
            "log_level": "ERROR",
        }

        # Act
        config = Config.from_dict(config_dict)

        # Assert
        assert config.gemini_api_key == "dict_key"
        assert config.persist_path == "/dict/path"
        assert config.embedding_dimension == 1024
        assert config.log_level == "ERROR"

    def test_from_dict_minimal(self):
        """Test creating config from minimal dictionary."""
        # Arrange
        config_dict = {"gemini_api_key": "minimal_dict_key"}

        # Act
        config = Config.from_dict(config_dict)

        # Assert
        assert config.gemini_api_key == "minimal_dict_key"
        assert config.persist_path == "./data/vector_store"  # Default
        assert config.embedding_dimension == 768  # Default
        assert config.log_level == "INFO"  # Default

    def test_get_config_function(self):
        """Test the get_config convenience function."""
        # Arrange
        env_vars = {"GEMINI_API_KEY": "function_test_key"}

        with patch.dict(os.environ, env_vars, clear=True):
            # Act
            config = get_config()

            # Assert
            assert isinstance(config, Config)
            assert config.gemini_api_key == "function_test_key"

    def test_config_validation(self):
        """Test Pydantic validation of config fields."""
        # Test that config accepts reasonable values
        config = Config(
            gemini_api_key="test", embedding_dimension=512  # Valid positive integer
        )
        assert config.embedding_dimension == 512
