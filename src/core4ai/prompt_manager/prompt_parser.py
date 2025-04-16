"""
Parser for reading prompts from markdown files.

This module handles loading and parsing prompt templates defined in markdown files.
"""
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger("core4ai.prompt_parser")

def parse_prompt_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse a markdown prompt file.
    
    Args:
        file_path: Path to the markdown prompt file
        
    Returns:
        Dictionary with prompt data or None if parsing failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract prompt name - this is the source of truth
        name_match = re.search(r'^#\s+Prompt\s+Name:\s+(.+?)$', content, re.MULTILINE)
        if not name_match:
            logger.warning(f"No 'Prompt Name:' found in {file_path}. Using filename as fallback.")
            prompt_name = file_path.stem
            
            # Ensure name has _prompt suffix
            if not prompt_name.endswith('_prompt'):
                prompt_name = f"{prompt_name}_prompt"
        else:
            prompt_name = name_match.group(1).strip()
        
        # Extract template section
        template_match = re.search(r'##\s*Template\s*\n+(.*?)(?:\n+##|\Z)', 
                                 content, re.DOTALL | re.MULTILINE)
        
        if not template_match:
            logger.warning(f"No template section found in {file_path}")
            return None
        
        template = template_match.group(1).strip()
        
        # Extract tags section
        tags = {}
        tags_match = re.search(r'##\s*Tags\s*\n+(.*?)(?:\n+##|\Z)', 
                             content, re.DOTALL | re.MULTILINE)
        
        if tags_match:
            tags_text = tags_match.group(1).strip()
            
            # Parse tags
            for line in tags_text.split('\n'):
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    tags[key] = value
        
        # Extract variables from template
        variables = []
        for match in re.finditer(r'{{([^{}]+)}}', template):
            var_name = match.group(1).strip()
            if var_name not in variables:
                variables.append(var_name)
        
        # Extract type from prompt_name - this is now reliable since prompt_name is explicit
        if 'type' not in tags and '_prompt' in prompt_name:
            prompt_type = prompt_name.replace('_prompt', '')
            tags['type'] = prompt_type
        
        # Build complete prompt data
        prompt_data = {
            "name": prompt_name,
            "template": template,
            "variables": variables,
            "tags": tags
        }
        
        return prompt_data
    
    except Exception as e:
        logger.error(f"Error parsing prompt file {file_path}: {e}")
        return None

def find_prompt_files(directory: Path) -> List[Path]:
    """
    Find all markdown prompt files in a directory.
    
    Args:
        directory: Path to search for prompt files
        
    Returns:
        List of paths to prompt files
    """
    if not directory.exists():
        logger.warning(f"Prompt directory does not exist: {directory}")
        return []
    
    try:
        return list(directory.glob("*.md"))
    except Exception as e:
        logger.error(f"Error finding prompt files: {e}")
        return []

def load_prompts_from_directory(directory: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all prompts from a directory.
    
    Args:
        directory: Path to the prompt directory
        
    Returns:
        Dictionary of parsed prompts
    """
    prompt_files = find_prompt_files(directory)
    prompts = {}
    
    for file_path in prompt_files:
        prompt_data = parse_prompt_file(file_path)
        if prompt_data:
            prompts[prompt_data["name"]] = prompt_data
    
    return prompts