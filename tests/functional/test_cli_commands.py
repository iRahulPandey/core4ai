"""
Functional tests for CLI commands.
"""
import pytest
import sys
import json
import asyncio
from unittest.mock import patch, AsyncMock
from click.testing import CliRunner

from src.core4ai import __version__
from src.core4ai.cli.commands import cli

class TestCLICommands:
    """Test the CLI commands."""
    
    def test_version_command(self, cli_runner, config_file):
        """Test the version command."""
        result = cli_runner.invoke(cli, ["version"])
        
        # Verify command ran successfully
        assert result.exit_code == 0
        assert "Core4AI version:" in result.stdout
        assert __version__ in result.stdout
    
    def test_help_command(self, cli_runner):
        """Test the help command."""
        result = cli_runner.invoke(cli, ["--help"])
        
        # Verify help output
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "Commands:" in result.stdout
        assert "chat" in result.stdout
        assert "register" in result.stdout
        assert "list" in result.stdout
    
    def test_register_samples_command(self, cli_runner, mock_mlflow, config_file):
        """Test registering sample prompts."""
        # Mock register_sample_prompts to return success
        with patch('src.core4ai.cli.commands.register_sample_prompts') as mock_register:
            mock_register.return_value = {
                "status": "success",
                "registered": 4,
                "results": [
                    {"name": "essay_prompt", "status": "success"},
                    {"name": "email_prompt", "status": "success"},
                    {"name": "technical_prompt", "status": "success"},
                    {"name": "creative_prompt", "status": "success"}
                ]
            }
            
            result = cli_runner.invoke(cli, ["register", "--samples"])
            
            # Verify command ran successfully
            assert result.exit_code == 0
            assert mock_register.called
            assert "success" in result.stdout
    
    def test_list_command(self, cli_runner, mock_mlflow, config_file):
        """Test listing prompts."""
        # Mock list_prompts to return sample data
        with patch('src.core4ai.cli.commands.registry_list_prompts') as mock_list:
            mock_list.return_value = {
                "status": "success",
                "prompts": [
                    {"name": "essay_prompt", "type": "essay", "variables": ["topic"], "latest_version": 1},
                    {"name": "email_prompt", "type": "email", "variables": ["formality", "recipient_type", "topic"], "latest_version": 1}
                ],
                "count": 2
            }
            
            result = cli_runner.invoke(cli, ["list"])
            
            # Verify command ran successfully
            assert result.exit_code == 0
            assert mock_list.called
            assert "essay_prompt" in result.stdout
            assert "email_prompt" in result.stdout
    
    def test_chat_command_simple(self, cli_runner, config_file):
        """Test chat command with simple output."""
        query = "Write an essay about AI"
        
        # Create a mock result that matches what process_query would return
        mock_result = {
            "original_query": query,
            "prompt_match": {"status": "matched", "prompt_name": "test_prompt", "confidence": 90},
            "content_type": "essay",
            "enhanced": True,
            "enhanced_query": "Enhanced query about AI",
            "validation_result": "VALID",
            "validation_issues": [],
            "response": "This is a mock response to the query."
        }
        
        # Instead of trying to mock the async function, mock get_provider_config
        # so it returns a valid config, avoiding real API calls
        with patch('src.core4ai.config.config.get_provider_config') as mock_get_config:
            # Provide a fake provider config
            mock_get_config.return_value = {
                "type": "openai",
                "api_key": "fake-key",
                "model": "gpt-3.5-turbo"
            }
            
            # Also mock process_query at the highest level
            with patch('src.core4ai.cli.commands.process_query') as mock_process:
                # Set up the mock to return our predefined result
                mock_process.return_value = mock_result
                
                # Run the command with simple flag
                result = cli_runner.invoke(cli, ["chat", "--simple", query])
                
                if result.exception:
                    print(f"Exception: {result.exception}")
                    print(f"Traceback: {result.exc_info}")
                
                # Verify command ran successfully
                assert result.exit_code == 0
                assert mock_process.called
                assert "This is a mock response to the query." in result.stdout
                
                # Simple output should not include metadata
                assert "Original Query:" not in result.stdout
                assert "Enhanced Query:" not in result.stdout

    def test_chat_command_detailed(self, cli_runner, config_file):
        """Test chat command with detailed output."""
        query = "Write an essay about AI"
        
        # Create a mock result that matches what process_query would return
        mock_result = {
            "original_query": query,
            "prompt_match": {"status": "matched", "prompt_name": "test_prompt", "confidence": 90},
            "content_type": "essay",
            "enhanced": True,
            "enhanced_query": "Enhanced query about AI",
            "validation_result": "VALID",
            "validation_issues": [],
            "response": "This is a mock response to the query."
        }
        
        # Mock get_provider_config
        with patch('src.core4ai.config.config.get_provider_config') as mock_get_config:
            # Provide a fake provider config
            mock_get_config.return_value = {
                "type": "openai",
                "api_key": "fake-key",
                "model": "gpt-3.5-turbo"
            }
            
            # Also mock process_query at the highest level
            with patch('src.core4ai.cli.commands.process_query') as mock_process:
                # Set up the mock to return our predefined result
                mock_process.return_value = mock_result
                
                # Run the command with default output (detailed)
                result = cli_runner.invoke(cli, ["chat", query])
                
                if result.exception:
                    print(f"Exception: {result.exception}")
                    print(f"Traceback: {result.exc_info}")
                
                # Verify command ran successfully
                assert result.exit_code == 0
                assert mock_process.called
                assert "This is a mock response to the query." in result.stdout

    def test_chat_command_verbose(self, cli_runner, config_file):
        """Test chat command with verbose flag."""
        query = "Write an essay about AI"
        
        # Create a mock result that matches what process_query would return
        mock_result = {
            "original_query": query,
            "prompt_match": {"status": "matched", "prompt_name": "test_prompt", "confidence": 90},
            "content_type": "essay",
            "enhanced": True,
            "enhanced_query": "Enhanced query about AI",
            "validation_result": "VALID",
            "validation_issues": [],
            "response": "This is a mock response to the query."
        }
        
        # Mock get_provider_config
        with patch('src.core4ai.config.config.get_provider_config') as mock_get_config:
            # Provide a fake provider config
            mock_get_config.return_value = {
                "type": "openai",
                "api_key": "fake-key",
                "model": "gpt-3.5-turbo"
            }
            
            # Also mock process_query at the highest level
            with patch('src.core4ai.cli.commands.process_query') as mock_process:
                # Set up the mock to return our predefined result
                mock_process.return_value = mock_result
                
                # Run the command with verbose flag
                result = cli_runner.invoke(cli, ["chat", "--verbose", query])
                
                if result.exception:
                    print(f"Exception: {result.exception}")
                    print(f"Traceback: {result.exc_info}")
                
                # Verify command ran successfully
                assert result.exit_code == 0
                assert mock_process.called

    def test_chat_command_provider_override(self, cli_runner, config_file):
        """Test chat command with provider override."""
        query = "Write an essay about AI"
        
        # Create a mock result that matches what process_query would return
        mock_result = {
            "original_query": query,
            "prompt_match": {"status": "matched", "prompt_name": "test_prompt", "confidence": 90},
            "content_type": "essay",
            "enhanced": True,
            "enhanced_query": "Enhanced query about AI", 
            "validation_result": "VALID",
            "validation_issues": [],
            "response": "This is a mock response to the query."
        }
        
        # Mock get_provider_config
        with patch('src.core4ai.config.config.get_provider_config') as mock_get_config:
            # Provide a fake provider config
            mock_get_config.return_value = {
                "type": "openai",
                "api_key": "fake-key",
                "model": "gpt-3.5-turbo"
            }
            
            # Set environment variable to avoid test errors
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                # Also mock process_query at the highest level
                with patch('src.core4ai.cli.commands.process_query') as mock_process:
                    # Set up the mock to return our predefined result
                    mock_process.return_value = mock_result
                    
                    # Run the command with provider override
                    result = cli_runner.invoke(cli, ["chat", "--provider", "openai", query])
                    
                    if result.exception:
                        print(f"Exception: {result.exception}")
                        print(f"Traceback: {result.exc_info}")
                    
                    # Verify command ran successfully
                    assert result.exit_code == 0
                    assert mock_process.called
    
    def test_register_command(self, cli_runner, mock_mlflow, config_file):
        """Test registering a prompt via CLI."""
        prompt = "Write a {{ length }} {{ content_type }} about {{ topic }}."
        
        # Mock register_prompt to return success
        with patch('src.core4ai.cli.commands.register_prompt') as mock_register:
            mock_register.return_value = {
                "name": "test_prompt",
                "status": "success",
                "version": 1
            }
            
            result = cli_runner.invoke(cli, [
                "register",
                "--name", "test_prompt",
                "--message", "Test prompt",
                prompt
            ])
            
            # Verify command ran successfully
            assert result.exit_code == 0
            assert mock_register.called
            
            # Check that register_prompt was called with correct args
            call_args = mock_register.call_args[1]
            assert call_args["name"] == "test_prompt"
            assert call_args["template"] == prompt
    
    def test_update_command(self, cli_runner, mock_mlflow, config_file):
        """Test updating a prompt via CLI."""
        prompt = "Updated {{ length }} {{ content_type }} about {{ topic }}."
        
        # Mock update_prompt to return success
        with patch('src.core4ai.cli.commands.update_prompt') as mock_update:
            mock_update.return_value = {
                "name": "test_prompt",
                "status": "success",
                "previous_version": 1,
                "new_version": 2
            }
            
            result = cli_runner.invoke(cli, [
                "update",
                "test_prompt",
                "--message", "Update test",
                prompt
            ])
            
            # Verify command ran successfully
            assert result.exit_code == 0
            assert mock_update.called
            
            # Check that update_prompt was called with correct args
            call_args = mock_update.call_args[1]
            assert call_args["name"] == "test_prompt"
            assert call_args["template"] == prompt