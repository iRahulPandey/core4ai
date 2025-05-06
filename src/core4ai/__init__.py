"""
Core4AI: Contextual Optimization and Refinement Engine for AI
-------------------------------------------------------------

A package for transforming basic user queries into optimized LLM prompts
using MLflow Prompt Registry.
"""

__version__ = "1.3.0"

from .cli.commands import cli
from .config.config import load_config, save_config, get_mlflow_uri, get_provider_config
from .providers import AIProvider
from .engine.processor import process_query, list_prompts
from .api import Core4AI
from .config.config_manager import Config

__all__ = [
    "cli", 
    "load_config", 
    "save_config", 
    "get_mlflow_uri", 
    "get_provider_config",
    "AIProvider",
    "process_query",
    "list_prompts",
    "Core4AI"
]