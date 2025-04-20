"""
Unit tests for the provider modules.
"""
import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
import aiohttp

from src.core4ai.providers import AIProvider
from src.core4ai.providers.openai_provider import OpenAIProvider
from src.core4ai.providers.ollama_provider import OllamaProvider
from langchain_ollama import ChatOllama  
from langchain_openai import ChatOpenAI 

class TestProviders:
    """Test the AI provider functionality."""
    
    def test_provider_factory(self):
        """Test creating providers using the factory method."""
        # Test creating OpenAI provider
        with patch('src.core4ai.providers.openai_provider.OpenAIProvider') as mock_openai:
            provider = AIProvider.create({"type": "openai", "api_key": "test-key"})
            assert provider == mock_openai.return_value
            # Updated to use keyword arguments
            mock_openai.assert_called_once_with(api_key="test-key", model="gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_openai_provider(self):
        """Test OpenAI provider functionality."""
        # Mock the ChatOpenAI class
        with patch('src.core4ai.providers.openai_provider.ChatOpenAI') as mock_chat:
            # Setup the mock
            mock_instance = mock_chat.return_value
            mock_instance.ainvoke = AsyncMock()
            mock_instance.ainvoke.return_value.content = "OpenAI response"
            
            # Create provider and test
            provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
            response = await provider.generate_response("Test prompt")
            
            # Verify results
            assert response == "OpenAI response"
            # No longer checking for temperature parameter
            mock_chat.assert_called_once_with(api_key="test-key", model="gpt-3.5-turbo")
    
    @pytest.mark.asyncio
    async def test_ollama_provider(self):
        """Test Ollama provider functionality."""
        # Mock the ChatOllama class
        with patch('src.core4ai.providers.ollama_provider.ChatOllama') as mock_chat:
            # Setup the mock
            mock_instance = mock_chat.return_value
            mock_instance.ainvoke = AsyncMock()
            mock_instance.ainvoke.return_value.content = "Ollama response"
            
            # Create provider and test
            provider = OllamaProvider(uri="http://test-uri", model="test-model")
            response = await provider.generate_response("Test prompt")
            
            # Verify results
            assert response == "Ollama response"
            mock_chat.assert_called_once_with(base_url="http://test-uri", model="test-model")

    @pytest.mark.asyncio
    async def test_openai_provider_error(self):
        """Test OpenAI provider error handling."""
        # Mock the ChatOpenAI class to raise exception
        with patch('src.core4ai.providers.openai_provider.ChatOpenAI') as mock_chat:
            # Setup the mock to raise
            mock_instance = mock_chat.return_value
            mock_instance.ainvoke = AsyncMock(side_effect=Exception("API error"))
            
            # Create provider and test
            provider = OpenAIProvider(api_key="test-key")
            
            # Now we expect the exception to be raised
            with pytest.raises(Exception, match="API error"):
                await provider.generate_response("Test prompt")
    
    @pytest.mark.asyncio
    async def test_openai_provider_error(self):
        """Test OpenAI provider error handling."""
        # Mock the ChatOpenAI class to raise exception
        with patch('src.core4ai.providers.openai_provider.ChatOpenAI') as mock_chat:
            # Setup the mock to raise
            mock_instance = mock_chat.return_value
            mock_instance.ainvoke = AsyncMock(side_effect=Exception("API error"))
            
            # Create provider and test
            provider = OpenAIProvider("test-key")
            response = await provider.generate_response("Test prompt")
            
            # Should return error message
            assert "Error generating response" in response