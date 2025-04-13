"""
Core4AI: Contextual Optimization and Refinement Engine for AI
-------------------------------------------------------------

A package for transforming basic user queries into optimized LLM prompts
using MLflow Prompt Registry.
"""

__version__ = "1.0.0"

from .cli.commands import cli
from .config.config import load_config, save_config, get_mlflow_uri, get_provider_config
from .providers import AIProvider

__all__ = [
    "cli", 
    "load_config", 
    "save_config", 
    "get_mlflow_uri", 
    "get_provider_config",
    "AIProvider"
]