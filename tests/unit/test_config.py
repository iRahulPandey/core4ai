"""
Unit tests for the configuration module.
"""
import os
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch

from src.core4ai.config.config import (
    load_config, save_config, get_mlflow_uri, get_provider_config
)

class TestConfig:
    """Test configuration loading and saving."""
    
    def test_save_and_load_config(self, temp_config_dir):
        """Test saving and loading a configuration."""
        # Create test config
        test_config = {
            "mlflow_uri": "http://test-mlflow:5000",
            "provider": {
                "type": "openai",
                "model": "test-model"
            }
        }
        
        # Save the config
        save_config(test_config)
        
        # Load the config
        loaded_config = load_config()
        
        # Verify it matches
        assert loaded_config["mlflow_uri"] == test_config["mlflow_uri"]
        assert loaded_config["provider"]["type"] == test_config["provider"]["type"]
        assert loaded_config["provider"]["model"] == test_config["provider"]["model"]
    
    def test_get_mlflow_uri(self, config_file):
        """Test getting MLflow URI from config."""
        uri = get_mlflow_uri()
        assert uri == "http://localhost:8080"
        
        # Test environment variable override
        os.environ["MLFLOW_TRACKING_URI"] = "http://env-override:5000"
        uri = get_mlflow_uri()
        assert uri == "http://env-override:5000"
        
        # Clean up
        del os.environ["MLFLOW_TRACKING_URI"]
    
    def test_get_provider_config(self, config_file):
        """Test getting provider configuration."""
        # Create mock config directly for this test to ensure 'model' is included
        test_config = {
            "mlflow_uri": "http://localhost:8080",
            "provider": {
                "type": "openai",
                "api_key": "test-openai-key-for-testing-only",
                "model": "gpt-3.5-turbo"
            }
        }
        
        # Use the temp directory from the fixture
        config_dir = os.path.dirname(config_file)
        
        # Save directly to ensure correct content
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(test_config, f)
        
        # Use patch.dict to ensure correct config dir is used
        with patch.dict('os.environ', {'CORE4AI_CONFIG_DIR': config_dir}):
            # Now test the function
            provider_config = get_provider_config()
            assert provider_config["type"] == "openai"
            assert provider_config["api_key"] == "test-openai-key-for-testing-only"
            assert provider_config["model"] == "gpt-3.5-turbo"
    
    def test_get_provider_config_with_ollama(self, ollama_config):
        """Test getting Ollama provider configuration."""
        provider_config = get_provider_config()
        assert provider_config["type"] == "ollama"
        assert provider_config["uri"] == "http://localhost:11434"
        assert provider_config["model"] == "llama2"
    
    def test_empty_config(self, temp_config_dir):
        """Test behavior with empty or missing config."""
        # Ensure config doesn't exist
        config_path = Path(temp_config_dir) / "config.yaml"
        if config_path.exists():
            config_path.unlink()
        
        # Override environment variable to use the temp directory
        with patch.dict('os.environ', {'CORE4AI_CONFIG_DIR': temp_config_dir}):
            # Load should return empty dict
            config = load_config()
            assert isinstance(config, dict)
            assert len(config) == 0
            
            # Provider config should have type=None
            provider_config = get_provider_config()
            assert provider_config["type"] is None
            
            # MLflow URI should be None
            assert get_mlflow_uri() is None