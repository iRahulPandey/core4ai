import os
import json
import logging
import re
import mlflow
from typing import Dict, Any, Optional, List

logger = logging.getLogger("core4ai.prompt_registry")

def setup_mlflow_connection():
    """Setup connection to MLflow server."""
    from ..config.config import get_mlflow_uri
    
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        raise ValueError("MLflow URI not configured. Run 'core4ai setup' first.")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"Using MLflow tracking URI: {mlflow_uri}")

def register_prompt(
    name: str, 
    template: str, 
    commit_message: str = "Initial commit", 
    tags: Optional[Dict[str, str]] = None, 
    version_metadata: Optional[Dict[str, str]] = None,
    set_as_production: bool = True
) -> Dict[str, Any]:
    """
    Register a prompt in MLflow Prompt Registry.
    
    Args:
        name: Name of the prompt
        template: Template text with variables in {{ variable }} format
        commit_message: Description of the prompt or changes
        tags: Optional key-value pairs for categorization
        version_metadata: Optional metadata for this prompt version
        set_as_production: Whether to set this version as the production alias
        
    Returns:
        Dictionary with registration details
    """
    setup_mlflow_connection()
    
    try:
        # Check if the prompt already exists with a production alias
        previous_production_version = None
        try:
            previous_prompt = mlflow.load_prompt(f"prompts:/{name}@production")
            previous_production_version = previous_prompt.version
            logger.info(f"Found existing production version {previous_production_version} for '{name}'")
        except Exception:
            logger.info(f"No existing production version found for '{name}'")
        
        # Register the prompt
        prompt = mlflow.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message,
            tags=tags or {},
            version_metadata=version_metadata or {}
        )
        
        # Handle aliasing
        if set_as_production:
            # Archive the previous production version if it exists
            if previous_production_version is not None:
                mlflow.set_prompt_alias(name, "archived", previous_production_version)
                logger.info(f"Archived '{name}' version {previous_production_version}")
                
            # Set new version as production
            mlflow.set_prompt_alias(name, "production", prompt.version)
            logger.info(f"Set '{name}' version {prompt.version} as production alias")
        
        result = {
            "name": name,
            "version": prompt.version,
            "status": "success",
            "production": set_as_production
        }
        
        # Add archived information if applicable
        if previous_production_version is not None:
            result["previous_production"] = previous_production_version
            result["archived"] = True
            
        return result
    except Exception as e:
        logger.error(f"Failed to register prompt '{name}': {e}")
        return {
            "name": name,
            "status": "error",
            "error": str(e)
        }

def register_from_file(file_path: str, set_as_production: bool = True) -> Dict[str, Any]:
    """
    Register prompts from a JSON file.
    
    The JSON file should have the format:
    {
        "prompts": [
            {
                "name": "prompt_name",
                "template": "Template text with {{ variables }}",
                "commit_message": "Description",
                "tags": {"key": "value"},
                "version_metadata": {"author": "name"}
            }
        ]
    }
    
    Args:
        file_path: Path to the JSON file
        set_as_production: Whether to set these versions as production aliases
        
    Returns:
        Dictionary with registration results
    """
    setup_mlflow_connection()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "prompts" not in data:
            raise ValueError("JSON file must contain a 'prompts' list")
        
        results = []
        for prompt_data in data["prompts"]:
            name = prompt_data.get("name")
            if not name:
                logger.warning("Skipping prompt without name")
                continue
                
            template = prompt_data.get("template")
            if not template:
                logger.warning(f"Skipping prompt '{name}' without template")
                continue
            
            result = register_prompt(
                name=name,
                template=template,
                commit_message=prompt_data.get("commit_message", "Registered from file"),
                tags=prompt_data.get("tags"),
                version_metadata=prompt_data.get("version_metadata"),
                set_as_production=set_as_production
            )
            results.append(result)
        
        return {
            "status": "success",
            "file": file_path,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to register prompts from file '{file_path}': {e}")
        return {
            "status": "error",
            "file": file_path,
            "error": str(e)
        }

def register_sample_prompts() -> Dict[str, Any]:
    """
    Register standard sample prompts for each content type.
    
    Returns:
        Dictionary with registration results
    """
    setup_mlflow_connection()
    
    results = []
    
    # Essay prompt
    essay_result = register_prompt(
        name="essay_prompt",
        template="""
        Write a well-structured essay on {{ topic }} that includes:
        - A compelling introduction that provides context and states your thesis
        - 2-3 body paragraphs, each with a clear topic sentence and supporting evidence
        - Logical transitions between paragraphs that guide the reader
        - A conclusion that synthesizes your main points and offers final thoughts
        
        The essay should be informative, well-reasoned, and demonstrate critical thinking.
        """,
        commit_message="Initial essay prompt",
        tags={"task": "writing", "type": "essay"}
    )
    results.append(essay_result)
    
    # Email prompt
    email_result = register_prompt(
        name="email_prompt",
        template="""
        Write a {{ formality }} email to my {{ recipient_type }} about {{ topic }} that includes:
        - A clear subject line
        - Appropriate greeting
        - Brief introduction stating the purpose
        - Main content in short paragraphs
        - Specific action items or requests clearly highlighted
        - Professional closing
        
        The tone should be {{ tone }}.
        """,
        commit_message="Initial email prompt",
        tags={"task": "writing", "type": "email"}
    )
    results.append(email_result)
    
    # Technical prompt
    technical_result = register_prompt(
        name="technical_prompt",
        template="""
        Provide a clear technical explanation of {{ topic }} for a {{ audience }} audience that:
        - Begins with a conceptual overview that anyone can understand
        - Uses analogies or real-world examples to illustrate complex concepts
        - Defines technical terminology when first introduced
        - Gradually increases in technical depth
        - Includes practical applications or implications where relevant
        - Addresses common misunderstandings or misconceptions
        """,
        commit_message="Initial technical prompt",
        tags={"task": "explanation", "type": "technical"}
    )
    results.append(technical_result)
    
    # Creative prompt
    creative_result = register_prompt(
        name="creative_prompt",
        template="""
        Write a creative {{ genre }} about {{ topic }} that:
        - Uses vivid sensory details and imagery
        - Develops interesting and multidimensional characters (if applicable)
        - Creates an engaging narrative arc with tension and resolution
        - Establishes a distinct mood, tone, and atmosphere
        - Employs figurative language to enhance meaning
        - Avoids clichés and predictable elements
        """,
        commit_message="Initial creative prompt",
        tags={"task": "writing", "type": "creative"}
    )
    results.append(creative_result)
    
    return {
        "status": "success",
        "registered": len(results),
        "results": results
    }

def load_all_prompts() -> Dict[str, Any]:
    """
    Load all available prompts from MLflow Prompt Registry.
    
    Returns:
        Dictionary mapping prompt names to their corresponding prompt objects
    """
    setup_mlflow_connection()
    
    prompts = {}
    
    # Known prompt naming patterns
    known_prompt_types = [
        "essay", "email", "technical", "creative", "code", 
        "summary", "analysis", "qa", "custom", "social_media", 
        "blog", "report", "letter", "presentation", "review",
        "comparison", "instruction"
    ]
    
    for prompt_type in known_prompt_types:
        prompt_name = f"{prompt_type}_prompt"
        
        # First try with production alias (preferred)
        try:
            prompt = mlflow.load_prompt(f"prompts:/{prompt_name}@production")
            logger.info(f"Loaded prompt '{prompt_name}' with production alias (version {prompt.version})")
            prompts[prompt_name] = prompt
            continue
        except Exception as e:
            logger.debug(f"Could not load prompt '{prompt_name}@production': {e}")
        
        # If production alias fails, try the latest version
        try:
            prompt = mlflow.load_prompt(f"prompts:/{prompt_name}")
            logger.info(f"Loaded latest version of prompt '{prompt_name}' (version {prompt.version})")
            prompts[prompt_name] = prompt
        except Exception as e:
            logger.debug(f"Could not load any version of prompt '{prompt_name}': {e}")
    
    logger.info(f"Loaded {len(prompts)} prompts from MLflow")
    return prompts

def list_prompts() -> Dict[str, Any]:
    """
    List all prompts in the MLflow Prompt Registry.
    
    Returns:
        Dictionary with prompt information
    """
    setup_mlflow_connection()
    
    try:
        # Standard content types
        content_types = ["essay", "email", "technical", "creative"]
        # Custom types we'll look for
        custom_types = ["code", "summary", "analysis", "qa", "custom", "social_media", 
                       "blog", "report", "letter", "presentation", "review", "comparison", 
                       "instruction"]
        # Combined list for checking
        all_types = content_types + custom_types
        
        prompts = []
        
        # Check for standard and custom prompt types
        for content_type in all_types:
            prompt_name = f"{content_type}_prompt"
            try:
                # Try different alias approaches to get as much information as possible
                production_version = None
                archived_version = None
                latest_prompt = None
                
                # Try to get production version
                try:
                    production_prompt = mlflow.load_prompt(f"prompts:/{prompt_name}@production")
                    production_version = production_prompt.version
                    latest_prompt = production_prompt  # Use production as latest if available
                except Exception:
                    pass
                
                # Try to get archived version
                try:
                    archived_prompt = mlflow.load_prompt(f"prompts:/{prompt_name}@archived")
                    archived_version = archived_prompt.version
                except Exception:
                    pass
                
                # If we don't have a production version, try to get latest
                if latest_prompt is None:
                    try:
                        latest_prompt = mlflow.load_prompt(f"prompts:/{prompt_name}")
                    except Exception:
                        continue  # Skip if we can't get any version
                
                # Extract variables from template
                variables = []
                for match in re.finditer(r'{{([^{}]+)}}', latest_prompt.template):
                    var_name = match.group(1).strip()
                    variables.append(var_name)
                
                # Add prompt information
                prompt_info = {
                    "name": prompt_name,
                    "type": content_type,
                    "latest_version": latest_prompt.version,
                    "production_version": production_version,
                    "archived_version": archived_version,
                    "variables": variables,
                    "tags": getattr(latest_prompt, "tags", {})
                }
                
                prompts.append(prompt_info)
            except Exception as e:
                # Skip if prompt doesn't exist or can't be loaded
                logger.debug(f"Could not load prompt '{prompt_name}': {e}")
        
        return {
            "status": "success",
            "prompts": prompts,
            "count": len(prompts)
        }
    except Exception as e:
        logger.error(f"Failed to list prompts: {e}")
        return {
            "status": "error",
            "error": str(e),
            "prompts": []
        }

def update_prompt(name: str, template: str, commit_message: str, set_as_production: bool = True) -> Dict[str, Any]:
    """
    Update an existing prompt with a new version.
    
    Args:
        name: Name of the prompt to update
        template: New template text
        commit_message: Description of the changes
        set_as_production: Whether to set this version as the production alias
        
    Returns:
        Dictionary with update details
    """
    setup_mlflow_connection()
    
    try:
        # Check if the prompt exists
        previous_version = None
        previous_production_version = None
        
        # Try to get the latest version
        try:
            previous_prompt = mlflow.load_prompt(f"prompts:/{name}")
            previous_version = previous_prompt.version
        except Exception as e:
            return {
                "name": name,
                "status": "error",
                "error": f"Prompt '{name}' not found: {str(e)}"
            }
        
        # Try to get the production version
        try:
            production_prompt = mlflow.load_prompt(f"prompts:/{name}@production")
            previous_production_version = production_prompt.version
        except:
            logger.info(f"No production alias found for '{name}'")
        
        # Register a new version
        prompt = mlflow.register_prompt(
            name=name,
            template=template,
            commit_message=commit_message
        )
        
        # Handle aliasing
        if set_as_production:
            # Archive the previous production version if it exists
            if previous_production_version is not None:
                mlflow.set_prompt_alias(name, "archived", previous_production_version)
                logger.info(f"Archived '{name}' version {previous_production_version}")
                
            # Set new version as production
            mlflow.set_prompt_alias(name, "production", prompt.version)
            logger.info(f"Set '{name}' version {prompt.version} as production alias")
        
        result = {
            "name": name,
            "previous_version": previous_version,
            "new_version": prompt.version,
            "status": "success",
            "production": set_as_production
        }
        
        # Add archived information if applicable
        if previous_production_version is not None:
            result["previous_production"] = previous_production_version
            result["archived"] = previous_production_version != prompt.version
            
        return result
    except Exception as e:
        logger.error(f"Failed to update prompt '{name}': {e}")
        return {
            "name": name,
            "status": "error",
            "error": str(e)
        }

def get_prompt_details(name: str) -> Dict[str, Any]:
    """
    Get detailed information about a prompt and all its versions.
    
    Args:
        name: Name of the prompt
        
    Returns:
        Dictionary with prompt details
    """
    setup_mlflow_connection()
    
    try:
        # Try to get production version
        production_version = None
        production_template = None
        try:
            production_prompt = mlflow.load_prompt(f"prompts:/{name}@production")
            production_version = production_prompt.version
            production_template = production_prompt.template
        except:
            pass
            
        # Try to get archived version
        archived_versions = []
        try:
            archived_prompt = mlflow.load_prompt(f"prompts:/{name}@archived")
            archived_versions.append(archived_prompt.version)
        except:
            pass
            
        # Try to get latest version
        latest_version = None
        latest_template = None
        latest_tags = None
        try:
            latest_prompt = mlflow.load_prompt(f"prompts:/{name}")
            latest_version = latest_prompt.version
            latest_template = latest_prompt.template
            latest_tags = getattr(latest_prompt, "tags", {})
        except Exception as e:
            return {
                "name": name,
                "status": "error",
                "error": f"Prompt '{name}' not found: {str(e)}"
            }
            
        # Extract variables from the template
        variables = []
        for match in re.finditer(r'{{([^{}]+)}}', latest_template):
            var_name = match.group(1).strip()
            variables.append(var_name)
            
        return {
            "name": name,
            "status": "success",
            "latest_version": latest_version,
            "production_version": production_version,
            "archived_versions": archived_versions,
            "variables": variables,
            "tags": latest_tags,
            "latest_template": latest_template,
            "production_template": production_template if production_version != latest_version else None
        }
    except Exception as e:
        logger.error(f"Failed to get details for prompt '{name}': {e}")
        return {
            "name": name,
            "status": "error",
            "error": str(e)
        }