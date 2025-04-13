# core4ai/server/server.py
import logging
import os
import asyncio
import sys
from mcp.server.fastmcp import FastMCP
from ..config.config import get_mlflow_uri, get_provider_config
from ..providers import AIProvider
from .workflow import create_workflow, load_all_prompts

logger = logging.getLogger("core4ai.server")

def start_server(mlflow_uri, provider_config):
    """
    Start the Core4AI server.
    
    Args:
        mlflow_uri: URI of the MLflow server
        provider_config: Configuration for the AI provider
    """
    # Set up environment variables
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    logger.info(f"Using MLflow tracking URI: {mlflow_uri}")
    
    # Initialize provider
    try:
        provider = AIProvider.create(provider_config)
        logger.info(f"Initialized AI provider: {provider_config.get('type')}")
    except Exception as e:
        logger.error(f"Failed to initialize provider: {e}")
        return
    
    # Initialize MCP server
    mcp = FastMCP(
        name="core4ai",
        instructions="AI query enhancement service that transforms basic queries into optimized prompts using MLflow Prompt Registry."
    )
    
    # Initialize workflow
    workflow_app = create_workflow(provider)
    
    @mcp.tool()
    async def optimize_query(query: str):
        """
        Optimize a user query by applying the appropriate template from MLflow.
        
        Args:
            query: The original user query
            
        Returns:
            Enhanced query and metadata
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Initialize state with available prompts
            prompts = load_all_prompts()
            
            # Run the workflow
            initial_state = {
                "user_query": query,
                "available_prompts": prompts
            }
            
            result = await workflow_app.ainvoke(initial_state)
            
            # Prepare response
            response = {
                "original_query": query,
                "prompt_match": result.get("prompt_match", {"status": "unknown"}),
                "content_type": result.get("content_type"),
                "enhanced": not result.get("should_skip_enhance", False),
                "initial_enhanced_query": result.get("enhanced_query", query) 
                    if result.get("validation_result") == "NEEDS_ADJUSTMENT" and not result.get("should_skip_enhance", False) 
                    else None,
                "enhanced_query": result.get("final_query") or result.get("enhanced_query", query),
                "validation_result": result.get("validation_result", "UNKNOWN"),
                "validation_issues": result.get("validation_issues", [])
            }
            
            logger.info(f"Successfully processed query")
            return response
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return {
                "original_query": query,
                "enhanced_query": query,  # Return original as fallback
                "enhanced": False,
                "error": str(e)
            }
    
    @mcp.tool()
    async def list_prompts():
        """
        List all available prompts from MLflow Prompt Registry.
        
        Returns:
            List of prompts and their metadata
        """
        try:
            prompts_dict = load_all_prompts()
            
            # Convert to a list of metadata for client display
            prompts_list = []
            for name, prompt in prompts_dict.items():
                # Extract content type from name (e.g., "essay_prompt" -> "essay")
                content_type = name.replace("_prompt", "")
                
                # Get variables from template
                import re
                variables = []
                for match in re.finditer(r'{{([^{}]+)}}', prompt.template):
                    var_name = match.group(1).strip()
                    variables.append(var_name)
                
                prompts_list.append({
                    "name": name,
                    "type": content_type,
                    "version": prompt.version,
                    "variables": variables,
                    "tags": getattr(prompt, "tags", {})
                })
            
            return {
                "prompts": prompts_list,
                "count": len(prompts_list)
            }
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return {
                "error": str(e),
                "prompts": [],
                "count": 0
            }
    
    # Run the server
    logger.info("Starting Core4AI server")
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)