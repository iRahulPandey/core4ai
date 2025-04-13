"""
Ollama provider for Core4AI.
"""
import logging
import aiohttp
from . import AIProvider

logger = logging.getLogger("core4ai.providers.ollama")

class OllamaProvider(AIProvider):
    """Ollama provider implementation."""
    
    def __init__(self, uri, model):
        """Initialize the Ollama provider with URI and model."""
        # Handle None URI case to prevent attribute errors
        if uri is None:
            logger.error("Ollama URI is None. Using default http://localhost:11434")
            self.uri = "http://localhost:11434"
        else:
            self.uri = uri.rstrip('/')
            
        self.model = model
        logger.info(f"Ollama provider initialized with model {model} at {self.uri}")
    
    async def generate_response(self, prompt):
        """Generate a response using Ollama."""
        url = f"{self.uri}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            logger.debug(f"Sending prompt to Ollama: {prompt[:50]}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama error: {error_text}")
                        return f"Error from Ollama: {error_text}"
                    
                    data = await response.json()
                    return data.get('response', '')
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            return f"Error generating response: {str(e)}"