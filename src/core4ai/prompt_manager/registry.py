"""
Registry module for Core4AI prompts.

This module handles registering, listing, and loading prompts from MLflow.
"""
import os
import json
import logging
import re
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

# Import new modules
from .prompt_parser import parse_prompt_file, find_prompt_files, load_prompts_from_directory
from .prompt_types import get_prompt_types, add_prompt_type, add_multiple_prompt_types, check_prompt_exists

logger = logging.getLogger("core4ai.prompt_registry")

# Default location for sample prompts
SAMPLE_PROMPTS_DIR = Path(__file__).parent.parent / "sample_prompts"

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
        # Initialize tags if None
        if tags is None:
            tags = {}
            
        # Extract type from name if not specified in tags
        if 'type' not in tags and '_prompt' in name:
            # Extract everything before _prompt as the type
            prompt_type = name.replace('_prompt', '')
            tags['type'] = prompt_type
            
            # Add to prompt types registry
            from .prompt_types import add_prompt_type
            add_prompt_type(prompt_type)
        elif 'type' in tags:
            # Add existing type to registry
            from .prompt_types import add_prompt_type
            add_prompt_type(tags['type'])
            
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
            tags=tags,
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
        prompt_types = []
        
        for prompt_data in data["prompts"]:
            name = prompt_data.get("name")
            if not name:
                logger.warning("Skipping prompt without name")
                continue
                
            template = prompt_data.get("template")
            if not template:
                logger.warning(f"Skipping prompt '{name}' without template")
                continue
            
            # Extract tags
            tags = prompt_data.get("tags", {})
            
            # Extract prompt type if available
            if 'type' in tags:
                prompt_types.append(tags['type'])
            elif '_' in name:
                # Try to derive type from name (e.g., essay_prompt -> essay)
                prompt_type = name.split('_')[0]
                prompt_types.append(prompt_type)
                if 'type' not in tags:
                    tags['type'] = prompt_type
            
            result = register_prompt(
                name=name,
                template=template,
                commit_message=prompt_data.get("commit_message", "Registered from file"),
                tags=tags,
                version_metadata=prompt_data.get("version_metadata"),
                set_as_production=set_as_production
            )
            results.append(result)
        
        # Register extracted prompt types
        if prompt_types:
            add_multiple_prompt_types(prompt_types)
        
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

def register_from_markdown(file_path: str, set_as_production: bool = True) -> Dict[str, Any]:
    """
    Register a prompt from a markdown file.
    
    Args:
        file_path: Path to the markdown file
        set_as_production: Whether to set this version as the production alias
        
    Returns:
        Dictionary with registration results
    """
    setup_mlflow_connection()
    
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return {
                "status": "error",
                "error": f"File '{file_path}' not found",
                "file": str(file_path)
            }
        
        # Parse the markdown file
        prompt_data = parse_prompt_file(file_path)
        if not prompt_data:
            return {
                "status": "error",
                "error": f"Failed to parse '{file_path}' as a prompt template",
                "file": str(file_path)
            }
        
        # Register the prompt
        result = register_prompt(
            name=prompt_data["name"],
            template=prompt_data["template"],
            commit_message=f"Registered from {file_path.name}",
            tags=prompt_data.get("tags", {}),
            set_as_production=set_as_production
        )
        
        # Add prompt type to registry if available
        if "type" in prompt_data.get("tags", {}):
            prompt_type = prompt_data["tags"]["type"]
            add_prompt_type(prompt_type)
        elif "_" in prompt_data["name"]:
            # Extract type from name (e.g., essay_prompt -> essay)
            prompt_type = prompt_data["name"].split("_")[0]
            add_prompt_type(prompt_type)
        
        return {
            **result,
            "file": str(file_path),
            "template_variables": prompt_data.get("variables", [])
        }
    
    except Exception as e:
        logger.error(f"Error registering prompt from '{file_path}': {e}")
        return {
            "status": "error",
            "error": str(e),
            "file": str(file_path)
        }

def register_sample_prompts(all_prompts=False, custom_dir=None, non_existing_only=False) -> Dict[str, Any]:
    """
    Register sample prompts from the sample prompts directory or a custom directory.
    
    Args:
        all_prompts: Whether to register all prompts regardless of existing versions
        custom_dir: Custom directory to load prompts from
        non_existing_only: Only register prompts that don't exist in MLflow
        
    Returns:
        Dictionary with registration results
    """
    setup_mlflow_connection()
    
    # Use custom directory if provided, otherwise use default
    prompts_dir = Path(custom_dir) if custom_dir else SAMPLE_PROMPTS_DIR
    
    # Ensure directory exists
    if not prompts_dir.exists():
        logger.warning(f"Prompt directory not found: {prompts_dir}")
        return {
            "status": "error",
            "error": f"Prompt directory not found: {prompts_dir}",
            "registered": 0,
            "results": []
        }
    
    # Get existing prompts if needed
    existing_prompts = set()
    if non_existing_only:
        try:
            existing_data = list_prompts()
            if existing_data["status"] == "success":
                existing_prompts = {p["name"] for p in existing_data["prompts"]}
        except Exception as e:
            logger.warning(f"Could not get existing prompts: {e}")
    
    # Load prompts from directory
    prompt_files = find_prompt_files(prompts_dir)
    results = []
    prompt_types = []
    
    for file_path in prompt_files:
        try:
            prompt_data = parse_prompt_file(file_path)
            if not prompt_data:
                continue
                
            prompt_name = prompt_data["name"]
            
            # Skip if exists and non_existing_only is True
            if non_existing_only and prompt_name in existing_prompts:
                logger.info(f"Skipping existing prompt: {prompt_name}")
                results.append({
                    "name": prompt_name,
                    "status": "skipped",
                    "message": "Prompt already exists"
                })
                continue
            
            # Register prompt
            result = register_prompt(
                name=prompt_name,
                template=prompt_data["template"],
                commit_message=f"Registered from {file_path.name}",
                tags=prompt_data.get("tags", {}),
                set_as_production=True
            )
            
            # Add prompt type to registry if available
            if "type" in prompt_data.get("tags", {}):
                prompt_type = prompt_data["tags"]["type"]
                prompt_types.append(prompt_type)
            elif "_" in prompt_name:
                # Extract type from name (e.g., essay_prompt -> essay)
                prompt_type = prompt_name.split("_")[0]
                prompt_types.append(prompt_type)
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error registering prompt from {file_path}: {e}")
            results.append({
                "name": file_path.stem,
                "status": "error",
                "error": str(e)
            })
    
    # Register all extracted prompt types
    if prompt_types:
        add_multiple_prompt_types(prompt_types)
    
    return {
        "status": "success",
        "registered": len([r for r in results if r.get("status") == "success"]),
        "skipped": len([r for r in results if r.get("status") == "skipped"]),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "total": len(results),
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
    
    # Get prompt types from registry
    prompt_types = get_prompt_types()
    
    # If no types in registry, use a default set
    if not prompt_types:
        prompt_types = [
            "essay", "email", "technical", "creative", "code", 
            "summary", "analysis", "qa", "custom", "social_media", 
            "blog", "report", "letter", "presentation", "review",
            "comparison", "instruction"
        ]
        
        # Add them to the registry for future use
        add_multiple_prompt_types(prompt_types)
    
    # Try to load each prompt type
    for prompt_type in prompt_types:
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
        # Get prompt types from registry
        prompt_types = get_prompt_types()
        
        # If no types in registry, use a default set
        if not prompt_types:
            prompt_types = [
                "essay", "email", "technical", "creative", "code", 
                "summary", "analysis", "qa", "custom", "social_media", 
                "blog", "report", "letter", "presentation", "review",
                "comparison", "instruction"
            ]
        
        prompts = []
        
        # Check for standard and custom prompt types
        for prompt_type in prompt_types:
            prompt_name = f"{prompt_type}_prompt"
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
                    "type": prompt_type,
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

def import_existing_prompts(prompt_names: List[str]) -> Dict[str, Any]:
    """
    Import existing prompts from MLflow into Core4AI.
    
    Args:
        prompt_names: List of prompt names to import
        
    Returns:
        Dictionary with import results
    """
    setup_mlflow_connection()
    
    results = []
    prompt_types = []
    
    for prompt_name in prompt_names:
        try:
            # Clean up the prompt name
            prompt_name = prompt_name.strip()
            if not prompt_name:
                continue
                
            # Try to load the prompt
            prompt = None
            try:
                # First try with production alias
                prompt = mlflow.load_prompt(f"prompts:/{prompt_name}@production")
                logger.info(f"Found prompt '{prompt_name}' with production alias")
            except Exception:
                try:
                    # Then try latest version
                    prompt = mlflow.load_prompt(f"prompts:/{prompt_name}")
                    logger.info(f"Found prompt '{prompt_name}' latest version")
                except Exception as e:
                    logger.warning(f"Could not find prompt '{prompt_name}': {e}")
                    results.append({
                        "name": prompt_name,
                        "status": "error",
                        "error": f"Could not find prompt in MLflow"
                    })
                    continue
            
            # Extract prompt type from name
            if "_" in prompt_name:
                prompt_type = prompt_name.split("_")[0]
                prompt_types.append(prompt_type)
                
            # Extract variables from template for info
            variables = []
            for match in re.finditer(r'{{([^{}]+)}}', prompt.template):
                var_name = match.group(1).strip()
                if var_name not in variables:
                    variables.append(var_name)
                
            results.append({
                "name": prompt_name,
                "status": "success",
                "version": prompt.version,
                "variables": variables
            })
            
        except Exception as e:
            logger.error(f"Error importing prompt '{prompt_name}': {e}")
            results.append({
                "name": prompt_name,
                "status": "error",
                "error": str(e)
            })
    
    # Register all extracted prompt types
    if prompt_types:
        add_multiple_prompt_types(prompt_types)
    
    return {
        "status": "success",
        "imported": len([r for r in results if r.get("status") == "success"]),
        "failed": len([r for r in results if r.get("status") == "error"]),
        "total": len(results),
        "results": results,
        "prompt_types": prompt_types
    }

def create_prompt_template(prompt_name: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create a template prompt file for users to customize.
    
    Args:
        prompt_name: Name of the prompt
        output_dir: Directory to save the template (defaults to current directory)
        
    Returns:
        Dictionary with creation results
    """
    if not output_dir:
        output_dir = Path.cwd()
    
    # Ensure directory exists
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure name has _prompt suffix
    if not prompt_name.endswith("_prompt"):
        prompt_name = f"{prompt_name}_prompt"
    
    # Get prompt type from name
    prompt_type = prompt_name.split("_")[0] if "_" in prompt_name else "custom"
    title = " ".join(word.capitalize() for word in prompt_type.split("_")) + " Prompt"
    
    # Create file path
    file_path = output_dir / f"{prompt_name}.md"
    
    # Create template content
    content = f"""# {title}

A template for {prompt_type} content.

## Template

Write a {{ style }} {prompt_type} about {{ topic }} that includes:
- Important point 1
- Important point 2
- Important point 3

Please ensure the tone is {{ tone }} and suitable for {{ audience }}.

## Metadata
- Type: {prompt_type}
- Task: writing
- Author: Custom
"""
    
    try:
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "file_path": str(file_path),
            "prompt_name": prompt_name
        }
    except Exception as e:
        logger.error(f"Error creating prompt template: {e}")
        return {
            "status": "error",
            "error": str(e),
            "prompt_name": prompt_name
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