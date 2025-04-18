"""
Query processor that leverages the workflow engine for prompt enhancement.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("core4ai.engine.processor")

async def process_query(query: str, provider_config: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Process a query through the Core4AI workflow.
    
    Args:
        query: The query to process
        provider_config: Optional provider configuration
        verbose: Whether to show verbose output
        
    Returns:
        Dict containing the processed query and response
    """
    from ..providers import AIProvider
    from .workflow import create_workflow
    
    # Important: Only fetch from config if not provided
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
        # Initialize provider with the provided configuration
        provider = AIProvider.create(provider_config)
        
        # Load prompts
        from ..prompt_manager.registry import load_all_prompts
        prompts = load_all_prompts()
        
        # Create workflow
        workflow = create_workflow()
        
        # Run workflow with provider config
        initial_state = {
            "user_query": query,
            "available_prompts": prompts,
            "provider_config": provider_config  # Pass provider config to workflow
        }
        
        if verbose:
            logger.info(f"Running workflow with query: {query}")
            logger.info(f"Using provider: {provider_config.get('type')}")
            logger.info(f"Using model: {provider_config.get('model', 'default')}")
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