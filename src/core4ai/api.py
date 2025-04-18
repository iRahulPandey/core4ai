"""
High-level Python API for Core4AI.

IMPORTANT NOTE ON MLFLOW PROMPTS:
When referencing specific prompts in MLflow, use aliases like @production.
For example: 'essay_prompt@production' instead of just 'essay_prompt'
This applies to both CLI commands and direct MLflow operations.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .config.config_manager import Config
from .engine.processor import process_query


class Core4AI:
    """
    High-level API for Core4AI.
    
    Examples:
        # Configure and use
        ai = Core4AI().configure_openai(api_key="your-key").set_mlflow_uri("http://localhost:8080")
        result = ai.chat("Write an essay about climate change")
        
        # Or use pre-configured settings
        ai = Core4AI() 
        result = ai.chat("Write an essay about climate change")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize Core4AI with optional configuration.
        
        Args:
            config: Configuration dictionary (if None, loads from file/env)
        """
        self.config = Config(load_existing=config is None)
        if config:
            self.config._config = config
    
    def set_mlflow_uri(self, uri: str) -> 'Core4AI':
        """
        Set MLflow URI.
        
        Args:
            uri: MLflow server URI
            
        Returns:
            Self for method chaining
        """
        self.config.set_mlflow_uri(uri)
        return self
    
    def configure_openai(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> 'Core4AI':
        """
        Configure OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: Model to use (default: gpt-3.5-turbo)
            
        Returns:
            Self for method chaining
        """
        self.config.use_openai(api_key, model)
        return self
    
    def configure_ollama(self, uri: str = "http://localhost:11434", model: str = "llama2") -> 'Core4AI':
        """
        Configure Ollama provider.
        
        Args:
            uri: Ollama server URI
            model: Ollama model to use
            
        Returns:
            Self for method chaining
        """
        self.config.use_ollama(uri, model)
        return self
    
    def save_config(self) -> 'Core4AI':
        """
        Save the current configuration.
        
        Returns:
            Self for method chaining
        """
        self.config.save()
        return self
    
    def register_samples(self) -> Dict[str, Any]:
        """
        Register sample prompts.
        
        Returns:
            Dictionary with registration results
        """
        return self.config.register_sample_prompts()
    
    def list_prompt_types(self) -> List[str]:
        """
        List all registered prompt types.
        
        Returns:
            List of prompt type names
        """
        return self.config.list_prompt_types()
    
    def add_prompt_type(self, prompt_type: str) -> 'Core4AI':
        """
        Add a new prompt type.
        
        Args:
            prompt_type: Name of the prompt type
            
        Returns:
            Self for method chaining
        """
        self.config.add_prompt_type(prompt_type)
        return self
    
    def list_prompts(self) -> Dict[str, Any]:
        """
        List available prompts.
        
        Returns:
            Dictionary with prompt information
        """
        return self.config.list_prompts()
    
    def create_prompt_template(self, prompt_name: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new prompt template.
        
        Args:
            prompt_name: Name of the prompt
            output_dir: Directory to save the template
            
        Returns:
            Dictionary with creation results
        """
        return self.config.create_prompt_template(prompt_name, output_dir)
    
    def register_prompt(self, name: str, template: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Register a single prompt.
        
        Args:
            name: Name of the prompt
            template: Template text
            tags: Optional tags
            
        Returns:
            Dictionary with registration results
        """
        return self.config.register_prompt(name, template, tags=tags)
    
    def register_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Register prompts from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with registration results
        """
        return self.config.register_from_file(file_path)
    
    def register_from_markdown(self, file_path: str) -> Dict[str, Any]:
        """
        Register a prompt from a markdown file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Dictionary with registration results
        """
        return self.config.register_from_markdown(file_path)
    
    def import_prompt_types(self, types: List[str]) -> 'Core4AI':
        """
        Import prompt types.
        
        Args:
            types: List of prompt type names
            
        Returns:
            Self for method chaining
        """
        self.config.import_prompt_types(types)
        return self
    
    def _show_missing_key_warning(self) -> None:
        """Display a warning message about missing OpenAI API key."""
        print("⚠️  OpenAI API key not found in environment variables.")
        print("Please export your OpenAI API key as OPENAI_API_KEY.")
        print("Example: export OPENAI_API_KEY='your-key-here'")
    
    def verify_openai_key(self) -> bool:
        """
        Verify that an OpenAI API key is properly configured.
        
        Returns:
            True if the key is available (either in config or environment)
        """
        provider_config = self.config.get_config().get('provider', {})
        
        if provider_config.get('type') == 'openai':
            # Check config first
            if provider_config.get('api_key'):
                return True
            
            # Then check environment
            if os.environ.get('OPENAI_API_KEY'):
                return True
                
            # Show warning message
            self._show_missing_key_warning()
            return False
        
        return True  # Not using OpenAI
    
    def chat(self, query: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Chat with AI using enhanced prompts.
        
        This method works in both regular Python scripts and notebooks
        with existing event loops.
        
        Args:
            query: The user query to process
            verbose: Whether to include verbose processing details
            
        Returns:
            Dictionary with response and processing details
        """
        # Get provider config
        provider_config = self._prepare_provider_config()
        
        # Check for OpenAI key if using OpenAI
        if provider_config.get('type') == 'openai' and not self._has_valid_openai_key(provider_config):
            return self._create_missing_key_response(query)
        
        # Execute the query with robust error handling
        try:
            return self._execute_async_query(query, provider_config, verbose)
        except Exception as e:
            # Provide a clean, user-friendly error message
            return {
                "error": f"Error processing query: {str(e)}",
                "original_query": query,
                "enhanced": False,
                "response": f"Error: {str(e)}"
            }
    
    def _prepare_provider_config(self) -> Dict[str, Any]:
        """
        Prepare the provider configuration with necessary keys from environment if needed.
        
        Returns:
            Updated provider configuration dictionary
        """
        provider_config = dict(self.config.get_config().get('provider', {}))
        
        # For OpenAI, ensure we have the API key from environment if not in config
        if provider_config.get('type') == 'openai' and not provider_config.get('api_key'):
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                provider_config['api_key'] = api_key
                
        return provider_config
    
    def _has_valid_openai_key(self, provider_config: Dict[str, Any]) -> bool:
        """
        Check if a valid OpenAI API key exists in the provided configuration.
        
        Args:
            provider_config: Provider configuration dictionary
            
        Returns:
            True if a valid key exists, False otherwise
        """
        return bool(provider_config.get('api_key'))
    
    def _create_missing_key_response(self, query: str) -> Dict[str, Any]:
        """
        Create a response object for missing OpenAI API key.
        
        Args:
            query: The original query
            
        Returns:
            Response dictionary with error information
        """
        # Show warning message
        self._show_missing_key_warning()
        
        return {
            "error": "OpenAI API key not found",
            "original_query": query,
            "enhanced": False,
            "response": "Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        }
    
    def _execute_async_query(self, query: str, provider_config: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        """
        Execute the query using the appropriate async approach based on environment.
        
        Args:
            query: The user query
            provider_config: Provider configuration
            verbose: Whether to show verbose output
            
        Returns:
            Response dictionary
        """
        # Check if we're in a Jupyter/IPython environment
        in_notebook = False
        try:
            # This will only succeed in IPython/Jupyter environments
            from IPython import get_ipython
            if get_ipython() is not None:
                in_notebook = True
                
                # Apply nest_asyncio for notebook environments
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    pass
        except ImportError:
            # Not in IPython/Jupyter
            pass
        
        try:
            if in_notebook:
                # We're in a notebook, use the current event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(process_query(query, provider_config, verbose))
            else:
                # Standard case - create new event loop
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                try:
                    # Use the newly created loop
                    return new_loop.run_until_complete(process_query(query, provider_config, verbose))
                finally:
                    # Properly close the loop
                    try:
                        # Cancel all running tasks
                        tasks = [t for t in asyncio.all_tasks(new_loop) if not t.done()]
                        if tasks:
                            # Create a gather task for all pending tasks with return_exceptions=True
                            new_loop.run_until_complete(
                                asyncio.gather(*tasks, return_exceptions=True)
                            )
                            
                        # Close the loop
                        new_loop.close()
                    except Exception:
                        # Suppress any errors during cleanup
                        pass
        except Exception as e:
            # Handle any exceptions
            return {
                "error": f"Error processing query: {str(e)}",
                "original_query": query,
                "enhanced": False,
                "response": f"Error: {str(e)}"
            }