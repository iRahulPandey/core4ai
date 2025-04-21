"""
Query processor that leverages the workflow engine for prompt enhancement.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("core4ai.engine.processor")

async def process_query(query: str, provider_config: Optional[Dict[str, Any]] = None, 
                      verbose: bool = False, record_analytics: bool = True) -> Dict[str, Any]:
    """
    Process a query through the Core4AI workflow.
    
    Args:
        query: The query to process
        provider_config: Optional provider configuration
        verbose: Whether to show verbose output
        record_analytics: Whether to record analytics data
        
    Returns:
        Dict containing the processed query and response
    """
    import time
    from ..providers import AIProvider
    from .workflow import create_workflow
    
    # Track start time for duration calculation
    start_time = time.time()
    
    # Important: Only fetch from config if not provided
    if not provider_config:
        from ..config.config import get_provider_config
        provider_config = get_provider_config()
    
    if not provider_config or not provider_config.get('type'):
        raise ValueError("AI provider not configured. Run 'core4ai setup' first.")
    
    # Ensure Ollama provider has a URI if type is ollama
    if provider_config.get('type') == 'ollama' and not provider_config.get('uri'):
        provider_config['uri'] = "http://localhost:11434"
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
        
        # Calculate duration
        duration = time.time() - start_time
        
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
            "response": result.get("response", "No response generated."),
            "duration": duration  # Add duration to response
        }
        
        # Record analytics if enabled
        if record_analytics:
            try:
                from ..analytics.tracking import record_prompt_usage
                
                prompt_match = response["prompt_match"]
                match_status = prompt_match.get("status", "unknown")
                
                if match_status == "matched":
                    # Get prompt details for matched prompt
                    prompt_name = prompt_match.get("prompt_name")
                    
                    # Try to get prompt version
                    try:
                        from ..prompt_manager.registry import get_prompt_details
                        prompt_details = get_prompt_details(prompt_name)
                        prompt_version = prompt_details.get("latest_version", 1)
                    except:
                        prompt_version = 1
                    
                    # Record the usage
                    record_prompt_usage(
                        prompt_name=prompt_name,
                        prompt_version=prompt_version,
                        confidence=prompt_match.get("confidence"),
                        duration=duration,
                        successful=True,  # Assume successful if we got this far
                        parameters=result.get("parameters"),
                        metadata={
                            "provider": provider_config.get("type"),
                            "model": provider_config.get("model"),
                            "enhanced": was_enhanced,
                            "needed_adjustment": needed_adjustment,
                            "fallback_used": prompt_match.get("fallback_used", False),
                            "content_type": result.get("content_type"),
                            "validation_result": result.get("validation_result"),
                            "validation_issues_count": len(result.get("validation_issues", [])),
                            "original_query_length": len(query),
                            "enhanced_query_length": len(enhanced_query) if enhanced_query else 0,
                            "response_length": len(response.get("response", ""))
                        }
                    )
                else:
                    # Record no match case
                    record_prompt_usage(
                        prompt_name="no_match",
                        prompt_version=1,
                        confidence=0,
                        duration=duration,
                        successful=True,
                        parameters={},
                        metadata={
                            "provider": provider_config.get("type"),
                            "model": provider_config.get("model"),
                            "match_status": match_status,
                            "original_query_length": len(query),
                            "response_length": len(response.get("response", ""))
                        }
                    )
                
                if verbose:
                    logger.info(f"Recorded analytics for query execution")
            except Exception as analytics_err:
                logger.warning(f"Failed to record analytics: {analytics_err}")
        
        # For logging validation issues when verbose
        if verbose and was_enhanced and needed_adjustment and response["validation_issues"]:
            for issue in response["validation_issues"]:
                logger.info(f"Validation issue: {issue}")
        
        return response
            
    except Exception as e:
        # Calculate duration even for errors
        duration = time.time() - start_time
        
        logger.error(f"Error processing query: {e}")
        error_response = {
            "error": str(e),
            "original_query": query,
            "enhanced": False,
            "response": f"Error processing query: {str(e)}"
        }
        
        # Record error in analytics
        if record_analytics:
            try:
                from ..analytics.tracking import record_prompt_usage
                
                record_prompt_usage(
                    prompt_name="error",
                    prompt_version=1,
                    confidence=0,
                    duration=duration,
                    successful=False,
                    parameters={},
                    metadata={
                        "provider": provider_config.get("type") if provider_config else None,
                        "model": provider_config.get("model") if provider_config else None,
                        "error": str(e),
                        "original_query_length": len(query)
                    }
                )
            except Exception as analytics_err:
                logger.warning(f"Failed to record analytics for error: {analytics_err}")
        
        return error_response

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