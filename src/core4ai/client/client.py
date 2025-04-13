import asyncio
import json
import logging
import os
import sys
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

# Set up logging
logger = logging.getLogger("core4ai.client")

async def process_query(query: str, provider_config: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Dict[str, Any]:
        """
        Process a query through the Core4AI workflow.
        
        Args:
            query: The query to process
            provider_config: Optional provider configuration (if not provided, will use from config)
            verbose: Whether to show verbose output
            
        Returns:
            Dict containing the processed query and response
        """
        from ..providers import AIProvider
        from ..server.workflow import create_workflow
        
        if not provider_config:
            from ..config.config import get_provider_config
            provider_config = get_provider_config()
        
        if not provider_config or not provider_config.get('type'):
            raise ValueError("AI provider not configured. Run 'core4ai setup' first.")
        
        # Ensure Ollama provider has a URI if type is ollama
        if provider_config.get('type') == 'ollama' and not provider_config.get('uri'):
            provider_config['uri'] = 'http://localhost:11434'
            logger.info(f"Using default Ollama URI: http://localhost:11434")
        
        try:
            # Initialize provider
            provider = AIProvider.create(provider_config)
            
            # Load prompts
            from ..prompt_manager.registry import load_all_prompts
            prompts = load_all_prompts()
            
            # Create workflow
            workflow = create_workflow()
            
            # Run workflow
            initial_state = {
                "user_query": query,
                "available_prompts": prompts
            }
            
            if verbose:
                logger.info(f"Running workflow with query: {query}")
                logger.info(f"Using provider: {provider_config.get('type')}")
                logger.info(f"Available prompts: {len(prompts)}")
            
            result = await workflow.ainvoke(initial_state)
            
            # Build response with complete enhancement traceability
            was_enhanced = not result.get("should_skip_enhance", False)
            needed_adjustment = result.get("validation_result") == "NEEDS_ADJUSTMENT"
            
            # Determine the enhanced and final queries
            enhanced_query = result.get("enhanced_query")
            final_query = result.get("final_query")
            
            response = {
                "original_query": query,
                "prompt_match": result.get("prompt_match", {"status": "unknown"}),
                "content_type": result.get("content_type"),
                "enhanced": was_enhanced,
                "initial_enhanced_query": enhanced_query if was_enhanced and needed_adjustment else None,
                "enhanced_query": final_query or enhanced_query or query,
                "validation_result": result.get("validation_result", "UNKNOWN"),
                "validation_issues": result.get("validation_issues", []),
                "response": result.get("response", "No response generated.")
            }
            
            # For logging validation issues when verbose
            if verbose and was_enhanced and needed_adjustment and response["validation_issues"]:
                for issue in response["validation_issues"]:
                    logger.info(f"Validation issue: {issue}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "original_query": query,
                "enhanced": False,
                "response": f"Error processing query: {str(e)}"
            }

def list_prompts() -> Dict[str, Any]:
    """
    List all available prompts.
    
    Returns:
        Dictionary with prompt information
    """
    try:
        from ..prompt_manager.registry import list_prompts as registry_list_prompts
        return registry_list_prompts()
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        return {
            "status": "error",
            "error": str(e),
            "prompts": []
        }

def verify_ollama_running(uri: str = "http://localhost:11434") -> bool:
    """
    Verify if Ollama is running at the given URI.
    
    Args:
        uri: The Ollama server URI
        
    Returns:
        True if Ollama is running, False otherwise
    """
    import requests
    
    try:
        response = requests.get(f"{uri}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def get_ollama_models(uri: str = "http://localhost:11434") -> list:
    """
    Fetch the list of available Ollama models.
    
    Args:
        uri: The Ollama server URI
        
    Returns:
        List of available model names
    """
    # Try using direct API call first
    import requests
    
    try:
        response = requests.get(f"{uri}/api/tags")
        if response.status_code == 200:
            data = response.json()
            models = [m.get('name') for m in data.get('models', [])]
            if models:
                return models
    except Exception as e:
        logger.debug(f"Failed to get Ollama models from API: {e}")
    
    # Fall back to CLI method
    FALLBACK_MODELS = ["llama2", "mistral", "gemma", "phi"]
    
    try:
        # Execute the Ollama list command
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        # Check if command executed successfully
        if result.returncode != 0:
            logger.warning(f"ollama list failed: {result.stderr}")
            return FALLBACK_MODELS
            
        # Parse the output to extract model names
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # Only header line or empty
            return FALLBACK_MODELS
            
        # Skip header line and extract the first column (model name)
        models = [line.split()[0] for line in lines[1:]]
        return models if models else FALLBACK_MODELS
        
    except (subprocess.SubprocessError, FileNotFoundError, IndexError) as e:
        logger.warning(f"Error fetching Ollama models: {str(e)}")
        return FALLBACK_MODELS