from abc import ABC, abstractmethod
import logging
from typing import Dict, Type, Any, Optional
from pydantic import BaseModel

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
    
    @property
    @abstractmethod
    def langchain_model(self):
        """Get the underlying LangChain model."""
        pass
    
    @abstractmethod
    def with_structured_output(self, output_schema: Type[BaseModel], method="function_calling"):
        """Get a version of the langchain model with structured output."""
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str, system_message: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """Generate a response for the given prompt with optional system message and temperature."""
        pass
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> 'AIProvider':
        """Factory method to create an AI provider based on configuration."""
        provider_type = config.get('type')
        
        if not provider_type:
            raise ValueError("Provider type not specified in configuration")
            
        provider_type = provider_type.lower()
        
        # Extract common parameters
        kwargs = {}
        for param in ['temperature', 'max_tokens', 'timeout', 'max_retries', 'organization', 'base_url']:
            if param in config:
                kwargs[param] = config[param]
        
        # Import providers dynamically to avoid circular imports
        if provider_type == 'openai':
            from .openai_provider import OpenAIProvider
            logger.info(f"Creating OpenAI provider with model {config.get('model', 'gpt-3.5-turbo')}")
            return OpenAIProvider(
                api_key=config.get('api_key'),
                model=config.get('model', "gpt-3.5-turbo"),
                temperature=config.get('temperature', 0.7),
                **kwargs
            )
        elif provider_type == 'ollama':
            from .ollama_provider import OllamaProvider
            logger.info(f"Creating Ollama provider with model {config.get('model')}")
            return OllamaProvider(
                uri=config.get('uri'),
                model=config.get('model'),
                temperature=config.get('temperature', 0.7),
                **kwargs
            )
        else:
            if provider_type not in cls._providers:
                raise ValueError(f"Unknown provider type: {provider_type}. Available types: {', '.join(cls._providers.keys())}")
            # Generic initialization for future providers
            provider_class = cls._providers[provider_type]
            return provider_class(config)