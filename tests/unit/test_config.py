"""
Minimal safe tests for the configuration module.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module itself, not the individual functions
from src.core4ai.config import config

class TestConfig:
    """Test configuration loading and saving."""
    
    def test_empty_config(self):
        """Test behavior with empty or missing config."""
        # Create a mock for load_config that returns an empty dict
        with patch('src.core4ai.config.config.load_config', return_value={}):
            # Test the functions directly
            assert config.get_mlflow_uri() is None
            assert config.get_provider_config()["type"] is None
    
    def test_config_with_values(self):
        """Test with mock config values."""
        test_config = {
            "mlflow_uri": "http://test-mlflow:5000",
            "provider": {
                "type": "openai",
                "api_key": "test-key",
                "model": "test-model"
            }
        }
        
        # Mock load_config to return our test values
        with patch('src.core4ai.config.config.load_config', return_value=test_config):
            # Test getting values from the mocked config
            uri = config.get_mlflow_uri()
            assert uri == "http://test-mlflow:5000"
            
            provider_config = config.get_provider_config()
            assert provider_config["type"] == "openai"
            assert provider_config["model"] == "test-model"
    
    def test_env_variables(self):
        """Test environment variables override config."""
        test_config = {
            "mlflow_uri": "http://from-config:5000",
            "provider": {
                "type": "openai",
                "api_key": "config-key",
                "model": "test-model"
            }
        }
        
        # Mock load_config and environment variables
        with patch('src.core4ai.config.config.load_config', return_value=test_config):
            with patch.dict('os.environ', {
                "MLFLOW_TRACKING_URI": "http://from-env:5000",
                "OPENAI_API_KEY": "env-key"
            }):
                # Environment should take precedence for MLflow URI
                uri = config.get_mlflow_uri()
                assert uri == "http://from-env:5000"
                
                # Test that env var takes precedence for API key
                provider_config = config.get_provider_config()
                assert provider_config["api_key"] == "env-key"
                assert provider_config["model"] == "test-model"