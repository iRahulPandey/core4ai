# core4ai/providers/__init__.py
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("core4ai.providers")

class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    async def generate_response(self, prompt):
        """Generate a response for the given prompt."""
        pass
    
    @classmethod
    def create(cls, config):
        """Factory method to create an AI provider based on configuration."""
        provider_type = config.get('type')
        
        if provider_type == 'openai':
            from .openai_provider import OpenAIProvider
            logger.info("Creating OpenAI provider")
            return OpenAIProvider(config.get('api_key'))
        
        elif provider_type == 'ollama':
            from .ollama_provider import OllamaProvider
            logger.info(f"Creating Ollama provider with model {config.get('model')}")
            return OllamaProvider(config.get('uri'), config.get('model'))
        
        raise ValueError(f"Unknown provider type: {provider_type}")