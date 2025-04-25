"""
High-level Python API for Core4AI.

IMPORTANT NOTE ON MLFLOW PROMPTS:
When referencing specific prompts in MLflow, use aliases like @production.
For example: 'essay_prompt@production' instead of just 'essay_prompt'
This applies to both CLI commands and direct MLflow operations.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .config.config_manager import Config
from .engine.processor import process_query
from .config.config import CONFIG_DIR, CONFIG_FILE


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
        # Load existing config if not provided
        self.config = Config(load_existing=config is None)
        
        if config:
            self.config._config = config
        else:
            # Check if config file exists
            if not CONFIG_FILE.exists():
                self._show_welcome_message()
            elif not self._check_config_complete():
                self._show_incomplete_config_message()
    
    def _check_config_complete(self) -> bool:
        """
        Check if the existing configuration has all required elements.
        
        Returns:
            True if configuration is complete, False otherwise
        """
        config_data = self.config.get_config()
        
        # Check for essential configuration items
        if not config_data.get('mlflow_uri'):
            return False
            
        provider = config_data.get('provider', {})
        if not provider.get('type'):
            return False
            
        return True
    
    def _show_welcome_message(self) -> None:
        """Display welcome message and setup instructions when no config exists."""
        print("\n🌟 Welcome to Core4AI! 🌟")
        print("No configuration file found at:", CONFIG_FILE)
        print("\nYou can configure Core4AI in two ways:")
        print("\n1️⃣ Using the CLI (recommended for first-time setup):")
        print("   $ core4ai setup")
        print("\n2️⃣ Using the Python API (current method):")
        print("   ai = Core4AI()")
        print("   ai.set_mlflow_uri('http://localhost:8080')")
        print("   ai.configure_openai(api_key='your-api-key')")  # or ai.configure_ollama()
        print("   ai.save_config()")
        print("\nConfiguration will be saved to:", CONFIG_FILE)
        print("\nℹ️  MLflow must be running to use Core4AI.")
        print("   Common MLflow URIs: http://localhost:5000, http://localhost:8080")
    
    def _show_incomplete_config_message(self) -> None:
        """Display message when configuration exists but is incomplete."""
        print("\n⚠️  Core4AI configuration is incomplete.")
        print("Configuration file exists at:", CONFIG_FILE)
        print("\nPlease ensure the following are configured:")
        print("1. MLflow URI: ai.set_mlflow_uri('http://localhost:8080')")
        print("2. AI Provider: ai.configure_openai() or ai.configure_ollama()")
        print("\nDon't forget to save your configuration with ai.save_config()")
    
    def set_mlflow_uri(self, uri: str) -> 'Core4AI':
        """
        Set MLflow URI.
        
        Args:
            uri: MLflow server URI
            
        Returns:
            Self for method chaining
        """
        self.config.set_mlflow_uri(uri)
        print(f"✅ MLflow URI set to: {uri}")
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
        # Store model and provider type in config
        self.config._config['provider'] = {
            'type': 'openai',
            'model': model
        }
        
        # Store API key in memory only, NOT in the config that will be saved
        self._openai_api_key = api_key
        
        # Provide feedback on configuration
        if api_key:
            if api_key.startswith('sk-') and len(api_key) >= 30:
                print(f"✅ OpenAI configured with provided API key and model: {model}")
                print("ℹ️ API key will be stored in memory only, not saved to config file")
            else:
                print("⚠️ OpenAI configured with invalid API key format. API calls may fail.")
        else:
            env_key = os.environ.get('OPENAI_API_KEY')
            if env_key:
                print(f"✅ OpenAI configured to use environment API key and model: {model}")
            else:
                print("⚠️ No API key provided or found in environment. API calls will fail.")
        
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
        print(f"✅ Ollama configured with URI: {uri} and model: {model}")
        return self
    
    def save_config(self) -> 'Core4AI':
        """
        Save the current configuration, excluding sensitive information.
        
        Returns:
            Self for method chaining
        """
        # Create a copy of the config to avoid modifying the original
        safe_config = dict(self.config._config)
        
        # Ensure we don't save API keys to disk
        if 'provider' in safe_config and safe_config['provider'].get('type') == 'openai':
            if 'api_key' in safe_config['provider']:
                del safe_config['provider']['api_key']
        
        # Save the sanitized config
        from .config.config import save_config as save_config_func
        save_config_func(safe_config)
        print(f"✅ Configuration saved securely (API keys excluded)")
        return self
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get a human-readable version of the current configuration.
        
        Returns:
            Dictionary with configuration information
        """
        config = self.config.get_config()
        provider = config.get('provider', {})
        
        readable_config = {
            "mlflow_uri": config.get('mlflow_uri', 'Not configured'),
            "provider_type": provider.get('type', 'Not configured'),
            "model": provider.get('model', 'default')
        }
        
        # Hide full API key but show if it exists
        if provider.get('type') == 'openai':
            api_key = provider.get('api_key')
            env_key = os.environ.get('OPENAI_API_KEY')
            
            if api_key:
                # Show only the first and last few characters
                if len(api_key) > 8:
                    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
                else:
                    masked_key = "[invalid key format]"
                readable_config["api_key"] = f"Provided ({masked_key})"
            elif env_key:
                if len(env_key) > 8:
                    masked_key = f"{env_key[:4]}...{env_key[-4:]}"
                else:
                    masked_key = "[invalid key format]"
                readable_config["api_key"] = f"From environment ({masked_key})"
            else:
                readable_config["api_key"] = "Not configured"
        
        print("Current Core4AI Configuration:")
        for key, value in readable_config.items():
            print(f"  {key}: {value}")
        
        return readable_config
    
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
    
    def _show_invalid_key_warning(self) -> None:
        """Display a warning message about invalid OpenAI API key."""
        print("⚠️  Invalid OpenAI API key provided.")
        print("OpenAI API keys should start with 'sk-' and be at least 30 characters long.")
        print("Please provide a valid API key or check your environment variable.")
    
    def verify_openai_key(self) -> bool:
        """
        Verify that an OpenAI API key is properly configured.
        
        Returns:
            True if the key is available and valid (either in config or environment)
        """
        provider_config = self.config.get_config().get('provider', {})
        
        if provider_config.get('type') == 'openai':
            provider_config = self._prepare_provider_config()
            # Check if we have a valid key format
            if self._has_valid_openai_key(provider_config):
                return True
                
            # Show warning message
            api_key = provider_config.get('api_key')
            if api_key:
                self._show_invalid_key_warning()
            else:
                self._show_missing_key_warning()
                
            return False
        
        return True  # Not using OpenAI
    
    def _prepare_provider_config(self) -> Dict[str, Any]:
        """
        Prepare the provider configuration with necessary keys from environment if needed.
        Only falls back to environment if no API key was explicitly set.
        
        Returns:
            Updated provider configuration dictionary
        """
        provider_config = dict(self.config.get_config().get('provider', {}))
        
        # For OpenAI, ensure we try to use the API key from environment ONLY if not in config
        if provider_config.get('type') == 'openai' and not provider_config.get('api_key'):
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                provider_config['api_key'] = api_key
                
        return provider_config
    
    def _has_valid_openai_key(self, provider_config: Dict[str, Any]) -> bool:
        """
        Check if a valid OpenAI API key exists in the provided configuration.
        A valid key should begin with 'sk-' and have sufficient length.
        
        Args:
            provider_config: Provider configuration dictionary
            
        Returns:
            True if a valid key exists, False otherwise
        """
        api_key = provider_config.get('api_key')
        
        # Basic validation that it's a potential OpenAI key
        if api_key and isinstance(api_key, str) and api_key.startswith('sk-') and len(api_key) >= 30:
            return True
            
        return False
    
    def _create_missing_key_response(self, query: str, key_exists: bool = False) -> Dict[str, Any]:
        """
        Create a response object for missing or invalid OpenAI API key.
        
        Args:
            query: The original query
            key_exists: Whether a key exists but is invalid (vs. missing entirely)
            
        Returns:
            Response dictionary with error information
        """
        if key_exists:
            # Show invalid key warning
            self._show_invalid_key_warning()
            error_msg = "Invalid OpenAI API key provided. API keys should start with 'sk-' and be at least 30 characters long."
        else:
            # Show missing key warning
            self._show_missing_key_warning()
            error_msg = "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        
        return {
            "error": "OpenAI API key issue",
            "original_query": query,
            "enhanced": False,
            "response": f"Error: {error_msg}"
        }
    
    def chat(self, query: str, verbose: bool = False, record_analytics: bool = True) -> Dict[str, Any]:
        """
        Chat with AI using enhanced prompts.
        
        This method works in both regular Python scripts and notebooks
        with existing event loops.
        
        Args:
            query: The user query to process
            verbose: Whether to include verbose processing details
            record_analytics: Whether to record analytics data
            
        Returns:
            Dictionary with response and processing details
        """
        # Check if configuration is complete
        if not self._check_config_complete():
            return {
                "error": "Incomplete configuration",
                "original_query": query,
                "enhanced": False,
                "response": "Error: Core4AI is not fully configured. Please set MLflow URI and configure an AI provider."
            }
        
        # Get in-memory provider config
        in_memory_provider = dict(self.config.get_config().get('provider', {}))
        provider_type = in_memory_provider.get('type', 'unknown')
        provider_model = in_memory_provider.get('model', 'default')
        
        # Add security-sensitive information to provider config (not saved to disk)
        if provider_type == 'openai':
            # First try in-memory key
            if hasattr(self, '_openai_api_key') and self._openai_api_key:
                in_memory_provider['api_key'] = self._openai_api_key
            # Then try environment variable
            elif os.environ.get('OPENAI_API_KEY'):
                in_memory_provider['api_key'] = os.environ.get('OPENAI_API_KEY')
                print("ℹ️ Using OpenAI API key from environment variable")
        
        # Show which provider is being used
        print(f"🔄 Processing query using {provider_type.upper()} provider with model: {provider_model}")
        
        # Check for OpenAI key if using OpenAI
        if provider_type == 'openai':
            api_key = in_memory_provider.get('api_key')
            
            # Check if we have a key but it's not valid
            if api_key and not self._has_valid_openai_key(in_memory_provider):
                return self._create_missing_key_response(query, key_exists=True)
            
            # Check if we have no key at all
            if not api_key:
                return self._create_missing_key_response(query, key_exists=False)
        
        # For Ollama, verify server connectivity
        if provider_type == 'ollama':
            from .providers.utilities import verify_ollama_running
            ollama_uri = in_memory_provider.get('uri', 'http://localhost:11434')
            if not verify_ollama_running(ollama_uri):
                return {
                    "error": "Ollama server not running",
                    "original_query": query,
                    "enhanced": False,
                    "response": f"Error: Could not connect to Ollama server at {ollama_uri}. Make sure Ollama is installed and running."
                }
            
            # Optionally, check if the specified model exists in Ollama
            if verbose:
                try:
                    from .providers.utilities import get_ollama_models
                    available_models = get_ollama_models(ollama_uri)
                    model_name = in_memory_provider.get('model')
                    if model_name not in available_models:
                        print(f"⚠️ Warning: Model '{model_name}' not found in Ollama. Available models: {', '.join(available_models)}")
                        print("   Ollama will attempt to pull the model or may use a default model.")
                except Exception as e:
                    print(f"ℹ️ Could not verify Ollama model: {str(e)}")
        
        # Execute the query with robust error handling
        try:
            # Pass the in-memory configuration to ensure consistency
            result = self._execute_async_query(query, in_memory_provider, verbose, record_analytics)
            
            # Add provider information to the response
            result["provider"] = {
                "type": provider_type,
                "model": provider_model
            }
            
            return result
        except Exception as e:
            # Create error response with provider information
            error_response = {
                "error": f"Error processing query: {str(e)}",
                "original_query": query,
                "enhanced": False,
                "response": f"Error: {str(e)}",
                "provider": {
                    "type": provider_type,
                    "model": provider_model
                }
            }
            return error_response
    
    def configure_analytics(self, enabled: bool = True, db_path: Optional[str] = None) -> 'Core4AI':
        """
        Configure analytics settings.
        
        Args:
            enabled: Whether analytics is enabled
            db_path: Path to the analytics database file (None for default)
            
        Returns:
            Self for method chaining
        """
        from .config.config import set_analytics_config
        
        # Update analytics config
        set_analytics_config(enabled=enabled, db_path=db_path)
        
        # Initialize analytics database if enabled
        if enabled:
            from .analytics.tracking import ensure_analytics_db
            if ensure_analytics_db():
                print(f"✅ Analytics enabled")
                if db_path:
                    print(f"Analytics database path: {db_path}")
            else:
                print(f"⚠️ Analytics initialization failed. Analytics will be disabled.")
        else:
            print(f"ℹ️ Analytics disabled")
            
        return self
    
    def get_prompt_analytics(self, prompt_name: Optional[str] = None, 
                       time_range: Optional[int] = None,
                       version: Optional[int] = None,
                       limit: int = 100) -> Dict[str, Any]:
        """
        Get analytics data for prompt usage.
        
        Args:
            prompt_name: Name of the prompt to analyze (None for all prompts)
            time_range: Time range in days (None for all time)
            version: Specific version to analyze (None for all versions)
            limit: Maximum number of records to return
            
        Returns:
            Dictionary with analytics data
        """
        from .analytics.tracking import get_prompt_analytics as get_analytics
        return get_analytics(prompt_name, time_range, version, limit)

    def get_usage_stats(self, time_range: Optional[int] = 30) -> Dict[str, Any]:
        """
        Get usage statistics for all prompts.
        
        Args:
            time_range: Time range in days (None for all time)
            
        Returns:
            Dictionary with usage statistics
        """
        from .analytics.tracking import get_usage_stats as get_stats
        return get_stats(time_range)

    def clear_analytics(self, prompt_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear analytics data.
        
        Args:
            prompt_name: Name of the prompt to clear data for (None for all prompts)
            
        Returns:
            Dictionary with operation status
        """
        from .analytics.tracking import clear_analytics as clear_data
        return clear_data(prompt_name)
    
    def _execute_async_query(self, query: str, provider_config: Dict[str, Any], 
                        verbose: bool, record_analytics: bool) -> Dict[str, Any]:
        """
        Execute the query using the appropriate async approach based on environment.
        
        Args:
            query: The user query
            provider_config: Provider configuration (will be used directly)
            verbose: Whether to show verbose output
            record_analytics: Whether to record analytics data
            
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
                # Pass provider_config directly to process_query
                return loop.run_until_complete(process_query(query, provider_config, verbose, record_analytics))
            else:
                # Standard case - create new event loop
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                
                try:
                    # Pass provider_config directly
                    return new_loop.run_until_complete(process_query(query, provider_config, verbose, record_analytics))
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
            
    def dashboard(self, output_dir: Optional[str] = None, 
              filename: Optional[str] = None, 
              time_range: Optional[int] = 30) -> str:
        """
        Generate an HTML dashboard with analytics data.
        
        Args:
            output_dir: Directory to save the dashboard (default: current directory)
            filename: Filename for the dashboard (default: coreai_stats_timestamp.html)
            time_range: Time range in days (default: 30)
            
        Returns:
            Path to the generated dashboard file
        """
        # Import the dashboard generator
        from .utils.dashboard import generate_dashboard
        
        # Get analytics data
        analytics_data = self.get_prompt_analytics(time_range=time_range)
        usage_data = self.get_usage_stats(time_range=time_range)
        
        # Check if analytics is enabled
        if analytics_data.get("status") == "error" or usage_data.get("status") == "error":
            if analytics_data.get("error", "").startswith("Analytics is disabled"):
                print("⚠️ Analytics is disabled. Enable it with: ai.configure_analytics(enabled=True)")
                return ""
            
            # Some other error occurred
            error_msg = analytics_data.get("error") or usage_data.get("error")
            print(f"⚠️ Error retrieving analytics data: {error_msg}")
            return ""
        
        # Generate the dashboard
        dashboard_path = generate_dashboard(
            analytics_data, 
            usage_data, 
            output_dir=output_dir,
            filename=filename
        )
        
        print(f"✅ Dashboard generated at: {dashboard_path}")
        
        # Try to automatically open the dashboard in browser
        try:
            import webbrowser
            webbrowser.open(f"file://{dashboard_path}")
        except Exception as e:
            print(f"Note: Could not open dashboard automatically. Open the file manually.")
        
        return dashboard_path