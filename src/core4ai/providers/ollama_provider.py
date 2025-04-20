"""
Ollama provider for Core4AI.
"""
import logging
from typing import Optional
from langchain_ollama import ChatOllama
from .base import AIProvider

logger = logging.getLogger("core4ai.providers.ollama")

class OllamaProvider(AIProvider):
    """Ollama provider implementation."""
    
    def __init__(self, uri=None, model=None, **kwargs):
        """Initialize the Ollama provider with URI and model."""
        # Handle None URI case to prevent attribute errors
        if uri is None:
            logger.warning("Ollama URI is None. Using default http://localhost:11434")
            self.uri = "http://localhost:11434"
        else:
            self.uri = uri.rstrip('/')
            
        self.model_name = model if model else "llama3.2:latest"
        
        # Build parameters dict with only non-None values
        model_params = {
            "base_url": self.uri,
            "model": self.model_name
        }
        
        # Only add optional parameters if they're not None
        for param in ['temperature', 'max_tokens', 'timeout', 'max_retries']:
            if param in kwargs and kwargs[param] is not None:
                model_params[param] = kwargs[param]
        
        # Use langchain-ollama's dedicated ChatOllama class
        self.model = ChatOllama(**model_params)
        
        logger.info(f"Ollama provider initialized with model {self.model_name} at {self.uri}")
    
    @property
    def langchain_model(self):
        """Get the underlying LangChain model."""
        return self.model
    
    async def generate_response(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate a response using Ollama."""
        try:
            logger.debug(f"Sending prompt to Ollama: {prompt[:50]}...")
            
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
            logger.error(f"Error generating response with Ollama: {e}")
            return f"Error generating response: {str(e)}"