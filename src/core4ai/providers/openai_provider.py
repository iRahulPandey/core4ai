"""
OpenAI provider for Core4AI.
"""
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from .base import AIProvider

logger = logging.getLogger("core4ai.providers.openai")

class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo", **kwargs):
        """Initialize the OpenAI provider with API key and model."""
        self.api_key = api_key
        self.model_name = model if model else "gpt-3.5-turbo"
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Responses may fail.")
        
        # Build parameters dict with only non-None values
        model_params = {
            "api_key": api_key,
            "model": self.model_name
        }
        
        # Only add optional parameters if they're not None
        for param in ['temperature', 'max_tokens', 'timeout', 'max_retries', 'organization', 'base_url']:
            if param in kwargs and kwargs[param] is not None:
                model_params[param] = kwargs[param]
        
        # Initialize with parameters matching the documentation
        self.model = ChatOpenAI(**model_params)
        
        logger.info(f"OpenAI provider initialized with model {self.model_name}")
    
    @property
    def langchain_model(self):
        """Get the underlying LangChain model."""
        return self.model
    
    async def generate_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate a response using OpenAI."""
        try:
            logger.debug(f"Sending prompt to OpenAI: {prompt[:50]}...")
            
            # Build messages array using the tuple format
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append(("system", system_message))
                
            # Add user message
            messages.append(("human", prompt))
            
            # Invoke the model asynchronously
            response = await self.model.ainvoke(messages)
            
            return response.content
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return f"Error generating response: {str(e)}"