"""
Registry for managing prompt types.

This module handles tracking and storing available prompt types.
"""
import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

logger = logging.getLogger("core4ai.prompt_types")

# Import config directory from config
from ..config.config import CONFIG_DIR

# File to store prompt types
PROMPT_TYPES_FILE = CONFIG_DIR / "prompt_types.json"

def get_prompt_types() -> List[str]:
    """
    Get the list of registered prompt types.
    
    Returns:
        List of prompt type names
    """
    if not PROMPT_TYPES_FILE.exists():
        return []
    
    try:
        with open(PROMPT_TYPES_FILE, 'r') as f:
            data = json.load(f)
            return data.get("types", [])
    except Exception as e:
        logger.error(f"Error loading prompt types: {e}")
        return []

def add_prompt_type(prompt_type: str) -> bool:
    """
    Add a new prompt type to the registry.
    
    Args:
        prompt_type: Name of the prompt type
    
    Returns:
        True if successful, False otherwise
    """
    # Make sure config directory exists
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    types = get_prompt_types()
    
    # Check if type already exists
    if prompt_type in types:
        return True
    
    # Add new type
    types.append(prompt_type)
    
    try:
        with open(PROMPT_TYPES_FILE, 'w') as f:
            json.dump({"types": types}, f)
        return True
    except Exception as e:
        logger.error(f"Error saving prompt types: {e}")
        return False

def add_multiple_prompt_types(prompt_types: List[str]) -> bool:
    """
    Add multiple prompt types to the registry.
    
    Args:
        prompt_types: List of prompt type names
    
    Returns:
        True if successful, False otherwise
    """
    # Make sure config directory exists
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    existing_types = set(get_prompt_types())
    
    # Add new types
    new_types = existing_types.union(set(prompt_types))
    
    try:
        with open(PROMPT_TYPES_FILE, 'w') as f:
            json.dump({"types": list(new_types)}, f)
        return True
    except Exception as e:
        logger.error(f"Error saving prompt types: {e}")
        return False

def remove_prompt_type(prompt_type: str) -> bool:
    """
    Remove a prompt type from the registry.
    
    Args:
        prompt_type: Name of the prompt type to remove
    
    Returns:
        True if successful, False otherwise
    """
    if not PROMPT_TYPES_FILE.exists():
        return False
    
    types = get_prompt_types()
    
    # Check if type exists
    if prompt_type not in types:
        return False
    
    # Remove type
    types.remove(prompt_type)
    
    try:
        with open(PROMPT_TYPES_FILE, 'w') as f:
            json.dump({"types": types}, f)
        return True
    except Exception as e:
        logger.error(f"Error saving prompt types: {e}")
        return False

def check_prompt_exists(prompt_name: str) -> bool:
    """
    Check if a prompt already exists in MLflow.
    
    Args:
        prompt_name: Name of the prompt to check
        
    Returns:
        True if it exists, False otherwise
    """
    import mlflow
    
    try:
        # Try to load the prompt (just check if it exists)
        try:
            mlflow.load_prompt(f"prompts:/{prompt_name}")
            return True
        except Exception:
            # Try with production alias
            try:
                mlflow.load_prompt(f"prompts:/{prompt_name}@production")
                return True
            except Exception:
                return False
    except Exception as e:
        logger.error(f"Error checking if prompt exists: {e}")
        return False