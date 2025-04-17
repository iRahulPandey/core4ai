"""
Shared pytest fixtures for Core4AI tests.

This file contains fixtures used across multiple test files.
"""
import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make sure core4ai package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import core4ai components
from src.core4ai.config.config import save_config
from src.core4ai.providers import AIProvider
from src.core4ai.prompt_manager.registry import register_prompt

# Define constants
TEST_MLFLOW_URI = "http://localhost:8080"
TEST_OPENAI_KEY = "test-openai-key-for-testing-only"
TEST_OLLAMA_URI = "http://localhost:11434"

# Mock for MLflow
class MockPrompt:
    """Mock for MLflow prompt objects."""
    def __init__(self, name, template, version=1, tags=None):
        self.name = name
        self.template = template
        self.version = version
        self.tags = tags or {}
    
    def format(self, **kwargs):
        """Format the prompt template with provided parameters."""
        result = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{{ {key} }}}}"
            result = result.replace(placeholder, str(value))
        return result

# Fixtures for configuration
@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configuration files."""
    temp_dir = tempfile.mkdtemp()
    old_env = dict(os.environ)
    
    # Set environment variable to use temp config dir
    os.environ["CORE4AI_CONFIG_DIR"] = temp_dir
    
    yield temp_dir
    
    # Cleanup
    os.environ.clear()
    os.environ.update(old_env)
    shutil.rmtree(temp_dir)

@pytest.fixture
def config_file(temp_config_dir):
    """Create a sample configuration file."""
    config = {
        "mlflow_uri": TEST_MLFLOW_URI,
        "provider": {
            "type": "openai",
            "api_key": TEST_OPENAI_KEY,
            "model": "gpt-3.5-turbo"  # Ensure model is included
        }
    }
    
    config_path = Path(temp_config_dir) / "config.yaml"
    save_config(config)
    
    return config_path

@pytest.fixture
def ollama_config(temp_config_dir):
    """Create a configuration with Ollama provider."""
    config = {
        "mlflow_uri": TEST_MLFLOW_URI,
        "provider": {
            "type": "ollama",
            "uri": TEST_OLLAMA_URI,
            "model": "llama2"
        }
    }
    
    save_config(config)
    return config

# Fixtures for mock providers
@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider for testing."""
    with patch('src.core4ai.providers.openai_provider.OpenAIProvider') as mock:
        provider = mock.return_value
        provider.generate_response.side_effect = lambda prompt: f"Mock response for: {prompt[:30]}..."
        yield provider

@pytest.fixture
def mock_ollama_provider():
    """Mock Ollama provider for testing."""
    with patch('src.core4ai.providers.ollama_provider.OllamaProvider') as mock:
        provider = mock.return_value
        provider.generate_response.side_effect = lambda prompt: f"Mock response for: {prompt[:30]}..."
        yield provider

# Fixtures for mock MLflow
@pytest.fixture
def mock_mlflow():
    """Mock MLflow interactions."""
    with patch('src.core4ai.prompt_manager.registry.mlflow') as mock_mlflow:
        # Setup load_prompt mock
        mock_prompts = {
            "essay_prompt": MockPrompt(
                "essay_prompt", 
                "Write a well-structured essay on {{ topic }} that includes an introduction, body, and conclusion.",
                version=1,
                tags={"type": "essay", "task": "writing"}
            ),
            "email_prompt": MockPrompt(
                "email_prompt",
                "Write a {{ formality }} email to my {{ recipient_type }} about {{ topic }}.",
                version=1,
                tags={"type": "email", "task": "writing"}
            ),
            "comparison_prompt": MockPrompt(
                "comparison_prompt",
                "Compare {{ item_1 }} and {{ item_2 }} in terms of {{ aspects }}.",
                version=1,
                tags={"type": "comparison", "task": "analysis"}
            )
        }
        
        def mock_load_prompt(prompt_name):
            """Mock function to load prompts."""
            if "@" in prompt_name:
                name = prompt_name.split("/")[1].split("@")[0]  # Extract name from "prompts:/name@alias"
            else:
                name = prompt_name.split("/")[1]  # Extract name from "prompts:/name"
                
            if name in mock_prompts:
                return mock_prompts[name]
            raise ValueError(f"Prompt {name} not found")
        
        mock_mlflow.load_prompt.side_effect = mock_load_prompt
        
        # Setup register_prompt mock
        def mock_register_prompt(name, template, commit_message="", tags=None, version_metadata=None):
            """Mock function to register prompts."""
            mock_prompts[name] = MockPrompt(
                name, 
                template, 
                version=len(mock_prompts) + 1,
                tags=tags or {}
            )
            return mock_prompts[name]
        
        mock_mlflow.register_prompt.side_effect = mock_register_prompt
        
        yield mock_mlflow, mock_prompts

# CLI testing fixtures
@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing commands."""
    from click.testing import CliRunner
    runner = CliRunner()
    return runner

@pytest.fixture
def mock_process_query():
    """Mock the process_query function for CLI testing."""
    # The correct import path must match the one used in cli/commands.py
    with patch('src.core4ai.engine.processor.process_query', autospec=True) as mock:
        async def mock_process(*args, **kwargs):
            return {
                "original_query": "test query",
                "prompt_match": {"status": "matched", "prompt_name": "test_prompt", "confidence": 90},
                "content_type": "essay",
                "enhanced": True,
                "initial_enhanced_query": "Enhanced test query",
                "enhanced_query": "Adjusted test query",
                "validation_result": "VALID",
                "validation_issues": [],
                "response": "This is a mock response to the query."
            }
        
        mock.side_effect = mock_process
        yield mock