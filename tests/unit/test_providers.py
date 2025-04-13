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

class TestProviders:
    """Test the AI provider functionality."""
    
    def test_provider_factory(self):
        """Test creating providers using the factory method."""
        # Test creating OpenAI provider
        with patch('src.core4ai.providers.openai_provider.OpenAIProvider') as mock_openai:
            provider = AIProvider.create({"type": "openai", "api_key": "test-key"})
            assert provider == mock_openai.return_value
            mock_openai.assert_called_once_with("test-key")
        
        # Test creating Ollama provider
        with patch('src.core4ai.providers.ollama_provider.OllamaProvider') as mock_ollama:
            provider = AIProvider.create({"type": "ollama", "uri": "test-uri", "model": "test-model"})
            assert provider == mock_ollama.return_value
            mock_ollama.assert_called_once_with("test-uri", "test-model")
        
        # Test invalid provider type
        with pytest.raises(ValueError):
            AIProvider.create({"type": "invalid"})
    
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
            provider = OpenAIProvider("test-key")
            response = await provider.generate_response("Test prompt")
            
            # Verify results
            assert response == "OpenAI response"
            mock_chat.assert_called_once_with(api_key="test-key", temperature=0.7)
            mock_instance.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ollama_provider(self):
        """Test Ollama provider functionality."""
        # Mock the JSON response
        json_response = {"response": "Ollama response"}
        
        # Create a context manager for ClientSession
        cm = MagicMock()
        cm.__aenter__ = AsyncMock()
        cm.__aexit__ = AsyncMock()
        
        # Create a context manager for the response
        resp_cm = MagicMock()
        resp_cm.__aenter__ = AsyncMock()
        resp_cm.__aexit__ = AsyncMock()
        
        # Set up the response attributes
        resp = AsyncMock()
        resp.status = 200
        resp.json = AsyncMock(return_value=json_response)
        resp_cm.__aenter__.return_value = resp
        
        # Set up the session
        session = AsyncMock()
        session.post = MagicMock(return_value=resp_cm)
        cm.__aenter__.return_value = session
        
        # Patch aiohttp.ClientSession
        with patch('aiohttp.ClientSession', return_value=cm):
            provider = OllamaProvider("http://test-uri", "test-model")
            response = await provider.generate_response("Test prompt")
            
            # Verify results
            assert response == "Ollama response"
            session.post.assert_called_once()
            url = session.post.call_args[0][0]
            assert url == "http://test-uri/api/generate"

    @pytest.mark.asyncio
    async def test_ollama_provider_error(self):
        """Test Ollama provider error handling."""
        # Create a context manager for ClientSession
        cm = MagicMock()
        cm.__aenter__ = AsyncMock()
        cm.__aexit__ = AsyncMock()
        
        # Create a context manager for the response
        resp_cm = MagicMock()
        resp_cm.__aenter__ = AsyncMock()
        resp_cm.__aexit__ = AsyncMock()
        
        # Set up the response attributes
        resp = AsyncMock()
        resp.status = 500
        resp.text = AsyncMock(return_value="Internal server error")
        resp_cm.__aenter__.return_value = resp
        
        # Set up the session
        session = AsyncMock()
        session.post = MagicMock(return_value=resp_cm)
        cm.__aenter__.return_value = session
        
        # Patch aiohttp.ClientSession
        with patch('aiohttp.ClientSession', return_value=cm):
            provider = OllamaProvider("http://test-uri", "test-model")
            response = await provider.generate_response("Test prompt")
            
            # Should return error message
            assert "Error from Ollama" in response
    
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