import logging

logger = logging.getLogger("core4ai.providers")

# Import the base AIProvider class
from .base import AIProvider

# Import utility functions (if utilities.py exists, otherwise we'll create it)
from .utilities import verify_ollama_running, get_ollama_models

# Export API
__all__ = ['AIProvider', 'verify_ollama_running', 'get_ollama_models']