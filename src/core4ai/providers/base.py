from abc import ABC, abstractmethod
import logging
from typing import Dict, Type, Any

logger = logging.getLogger("core4ai.providers.base")

class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    # Registry for provider classes
    _providers: Dict[str, Type['AIProvider']] = {}
    
    def __init_subclass__(cls, **kwargs):
        """Auto-register provider subclasses."""
        super().__init_subclass__(**kwargs)
        # Register the provider using the class name
        provider_type = cls.__name__.lower().replace('provider', '')
        AIProvider._providers[provider_type] = cls
        logger.debug(f"Registered provider: {provider_type}")
    
    @abstractmethod
    async def generate_response(self, prompt):
        """Generate a response for the given prompt."""
        pass
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> 'AIProvider':
        """Factory method to create an AI provider based on configuration."""
        provider_type = config.get('type')
        
        if not provider_type:
            raise ValueError("Provider type not specified in configuration")
            
        provider_type = provider_type.lower()
        
        # Import providers dynamically to avoid circular imports
        if provider_type == 'openai':
            from .openai_provider import OpenAIProvider
            logger.info(f"Creating OpenAI provider with model {config.get('model', 'gpt-3.5-turbo')}")
            return OpenAIProvider(
                config.get('api_key'), 
                config.get('model', "gpt-3.5-turbo")
            )
        elif provider_type == 'ollama':
            from .ollama_provider import OllamaProvider
            logger.info(f"Creating Ollama provider with model {config.get('model')}")
            return OllamaProvider(config.get('uri'), config.get('model'))
        else:
            if provider_type not in cls._providers:
                raise ValueError(f"Unknown provider type: {provider_type}. Available types: {', '.join(cls._providers.keys())}")
            # Generic initialization for future providers
            provider_class = cls._providers[provider_type]
            return provider_class(config)