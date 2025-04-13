"""
Unit tests for the prompt registry module.
"""
import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core4ai.prompt_manager.registry import (
    register_prompt,
    register_from_file,
    list_prompts,
    load_all_prompts,
    update_prompt,
    get_prompt_details
)

class TestPromptRegistry:
    """Test the prompt registry functionality."""
    
    def test_register_prompt(self, mock_mlflow, temp_config_dir):
        """Test registering a new prompt."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Register a test prompt
        result = register_prompt(
            name="test_prompt",
            template="This is a {{ test }} template",
            commit_message="Test commit",
            tags={"type": "test", "task": "testing"}
        )
        
        # Verify result
        assert result["status"] == "success"
        assert result["name"] == "test_prompt"
        assert "version" in result
        
        # Verify mock was called correctly
        mock_mlflow_obj.register_prompt.assert_called_once()
        call_args = mock_mlflow_obj.register_prompt.call_args[1]
        assert call_args["name"] == "test_prompt"
        assert call_args["template"] == "This is a {{ test }} template"
        assert call_args["tags"] == {"type": "test", "task": "testing"}
    
    def test_register_prompt_error(self, mock_mlflow, temp_config_dir):
        """Test error handling when registering a prompt."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Make the mock raise an error
        mock_mlflow_obj.register_prompt.side_effect = Exception("Test error")
        
        # Try to register a prompt
        result = register_prompt(
            name="error_prompt",
            template="This will {{ fail }}",
            commit_message="Error test"
        )
        
        # Verify error result
        assert result["status"] == "error"
        assert "error" in result
        assert result["name"] == "error_prompt"
    
    def test_register_from_file(self, mock_mlflow, temp_config_dir, tmp_path):
        """Test registering prompts from a file."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create a temporary prompts file
        prompts_file = tmp_path / "test_prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump({
                "prompts": [
                    {
                        "name": "file_prompt_1",
                        "template": "Template 1 with {{ var1 }}",
                        "commit_message": "From file 1",
                        "tags": {"type": "file", "task": "test1"}
                    },
                    {
                        "name": "file_prompt_2",
                        "template": "Template 2 with {{ var2 }}",
                        "commit_message": "From file 2",
                        "tags": {"type": "file", "task": "test2"}
                    }
                ]
            }, f)
        
        # Register from file
        result = register_from_file(str(prompts_file))
        
        # Verify result
        assert result["status"] == "success"
        assert result["count"] == 2
        assert len(result["results"]) == 2
        
        # Verify mock was called correctly
        assert mock_mlflow_obj.register_prompt.call_count == 2
    
    def test_register_from_file_error(self, mock_mlflow, temp_config_dir, tmp_path):
        """Test error handling with invalid prompt file."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Create an invalid file (not JSON)
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("Not valid JSON")
        
        # Try to register from invalid file
        result = register_from_file(str(invalid_file))
        
        # Verify error result
        assert result["status"] == "error"
        assert "error" in result
    
    def test_list_prompts(self, mock_mlflow, temp_config_dir):
        """Test listing all prompts."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Setup the mock prompts
        mock_mlflow_obj.load_prompt.side_effect = lambda x: mock_prompts.get(x.split("/")[1].split("@")[0])
        
        # List prompts
        result = list_prompts()
        
        # Verify result
        assert result["status"] == "success"
        assert "prompts" in result
        assert "count" in result
        assert isinstance(result["prompts"], list)
    
    def test_update_prompt(self, mock_mlflow, temp_config_dir):
        """Test updating an existing prompt."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # First register a prompt
        register_prompt(
            name="update_test",
            template="Original {{ template }}",
            commit_message="Original"
        )
        
        # Now update it
        result = update_prompt(
            name="update_test",
            template="Updated {{ template }}",
            commit_message="Updated version"
        )
        
        # Verify result
        assert result["status"] == "success"
        assert result["name"] == "update_test"
        assert "previous_version" in result
        assert "new_version" in result
    
    def test_get_prompt_details(self, mock_mlflow, temp_config_dir):
        """Test getting detailed information about a prompt."""
        mock_mlflow_obj, mock_prompts = mock_mlflow
        
        # Register a test prompt first
        register_prompt(
            name="detail_test",
            template="Template with {{ var1 }} and {{ var2 }}",
            commit_message="For details test",
            tags={"type": "detail", "task": "testing"}
        )
        
        # Get details
        result = get_prompt_details("detail_test")
        
        # Verify result
        assert result["status"] == "success"
        assert result["name"] == "detail_test"
        assert "latest_version" in result
        assert "variables" in result
        assert "var1" in result["variables"]
        assert "var2" in result["variables"]
        assert "latest_template" in result