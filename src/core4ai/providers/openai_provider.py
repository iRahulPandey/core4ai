# core4ai/providers/openai_provider.py
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from . import AIProvider

logger = logging.getLogger("core4ai.providers.openai")

class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key):
        """Initialize the OpenAI provider with API key."""
        self.api_key = api_key
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Responses may fail.")
        
        self.model = ChatOpenAI(api_key=api_key, temperature=0.7)
        logger.info("OpenAI provider initialized")
    
    async def generate_response(self, prompt):
        """Generate a response using OpenAI."""
        try:
            logger.debug(f"Sending prompt to OpenAI: {prompt[:50]}...")
            response = await self.model.ainvoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {e}")
            return f"Error generating response: {str(e)}"