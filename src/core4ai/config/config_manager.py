"""
Configuration manager for Core4AI.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
from .config import save_config, load_config, ensure_config_dir

class Config:
    """Configuration manager for Core4AI."""
    
    def __init__(self, load_existing: bool = True):
        """
        Initialize configuration object.
        
        Args:
            load_existing: Whether to load existing config (if available)
        """
        self._config = load_config() if load_existing else {}
    
    def set_mlflow_uri(self, uri: str) -> 'Config':
        """
        Set the MLflow URI.
        
        Args:
            uri: URI for the MLflow server
            
        Returns:
            Self for method chaining
        """
        self._config['mlflow_uri'] = uri
        return self
    
    def use_openai(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> 'Config':
        """
        Configure OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: Model to use (default: gpt-3.5-turbo)
            
        Returns:
            Self for method chaining
        """
        self._config['provider'] = {
            'type': 'openai',
            'model': model
        }
        if api_key:
            self._config['provider']['api_key'] = api_key
        return self
        
    def use_ollama(self, uri: str = "http://localhost:11434", model: str = "llama2") -> 'Config':
        """
        Configure Ollama provider.
        
        Args:
            uri: Ollama server URI
            model: Ollama model to use
            
        Returns:
            Self for method chaining
        """
        self._config['provider'] = {
            'type': 'ollama',
            'uri': uri,
            'model': model
        }
        return self
    
    def import_prompt_types(self, types: List[str]) -> 'Config':
        """
        Import prompt types.
        
        Args:
            types: List of prompt type names
            
        Returns:
            Self for method chaining
        """
        from ..prompt_manager.prompt_types import add_multiple_prompt_types
        add_multiple_prompt_types(types)
        return self
        
    def register_sample_prompts(self) -> Dict[str, Any]:
        """
        Register sample prompts.
        
        Returns:
            Dictionary with registration results
        """
        from ..prompt_manager.registry import register_sample_prompts
        return register_sample_prompts()
    
    def list_prompt_types(self) -> List[str]:
        """
        List all registered prompt types.
        
        Returns:
            List of prompt type names
        """
        from ..prompt_manager.prompt_types import get_prompt_types
        return get_prompt_types()
    
    def add_prompt_type(self, prompt_type: str) -> bool:
        """
        Add a new prompt type.
        
        Args:
            prompt_type: Name of the prompt type
            
        Returns:
            True if successful, False otherwise
        """
        from ..prompt_manager.prompt_types import add_prompt_type
        return add_prompt_type(prompt_type)
    
    def list_prompts(self) -> Dict[str, Any]:
        """
        List available prompts.
        
        Returns:
            Dictionary with prompt information
        """
        from ..prompt_manager.registry import list_prompts
        return list_prompts()
    
    def create_prompt_template(self, prompt_name: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new prompt template.
        
        Args:
            prompt_name: Name of the prompt
            output_dir: Directory to save the template
            
        Returns:
            Dictionary with creation results
        """
        from ..prompt_manager.registry import create_prompt_template
        return create_prompt_template(prompt_name, Path(output_dir) if output_dir else None)
    
    def register_prompt(self, name: str, template: str, commit_message: str = "Via Python API", 
                       tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Register a single prompt.
        
        Args:
            name: Name of the prompt
            template: Template text
            commit_message: Commit message
            tags: Optional tags
            
        Returns:
            Dictionary with registration results
        """
        from ..prompt_manager.registry import register_prompt
        return register_prompt(name, template, commit_message, tags)
    
    def register_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Register prompts from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with registration results
        """
        from ..prompt_manager.registry import register_from_file
        return register_from_file(file_path)
    
    def register_from_markdown(self, file_path: str) -> Dict[str, Any]:
        """
        Register a prompt from a markdown file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dictionary with registration results
        """
        from ..prompt_manager.registry import register_from_markdown
        return register_from_markdown(file_path)
    
    def save(self) -> None:
        """Save configuration to file."""
        save_config(self._config)
        
    def get_config(self) -> Dict[str, Any]:
        """Get the configuration dictionary."""
        return self._config