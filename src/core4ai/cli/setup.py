# core4ai/cli/setup.py
import os
import click
import requests
import logging
import subprocess
import time
from pathlib import Path
from ..config.config import load_config, save_config, ensure_config_dir, CONFIG_DIR
from ..providers.utilities import verify_ollama_running, get_ollama_models

logger = logging.getLogger("core4ai.setup")

def validate_mlflow_uri(uri):
    """Validate MLflow URI by attempting to connect."""
    # Try multiple MLflow endpoints to validate the connection
    endpoints = [
        "/api/2.0/mlflow/experiments/list",  # Standard REST API
        "/ajax-api/2.0/mlflow/experiments/list",  # Alternative path
        "/",  # Root path (at least check if the server responds)
    ]
    
    for endpoint in endpoints:
        try:
            # Try with trailing slash trimmed
            clean_uri = uri.rstrip('/')
            url = f"{clean_uri}{endpoint}"
            logger.debug(f"Trying to connect to MLflow at: {url}")
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to MLflow at {url}")
                return True
            else:
                logger.debug(f"Response from {url}: {response.status_code}")
        except Exception as e:
            logger.debug(f"Failed to connect to {endpoint}: {str(e)}")
    
    # If we get here, none of the endpoints worked
    logger.warning(f"Could not validate MLflow at {uri} on any standard endpoint")
    return False

# Add to src/core4ai/cli/setup.py

def setup_wizard():
    """Interactive setup wizard for core4ai."""
    click.echo("┌──────────────────────────────────────────────────────┐")
    click.echo("│             Core4AI Setup Wizard                     │")
    click.echo("│ Contextual Optimization and Refinement Engine for AI │")
    click.echo("└──────────────────────────────────────────────────────┘")
    
    click.echo("\nThis wizard will help you configure Core4AI for your environment.")
    
    # Initialize config
    config = load_config()
    previous_provider = config.get('provider', {}).get('type')
    
    # MLflow URI
    mlflow_uri = click.prompt(
        "Enter your MLflow URI",
        default=config.get('mlflow_uri', 'http://localhost:8080')
    )
    
    if not validate_mlflow_uri(mlflow_uri):
        click.echo("\n⚠️  Warning: Could not connect to MLflow at the provided URI.")
        click.echo("    Please ensure MLflow is running and accessible at this address.")
        click.echo("    Common MLflow URLs: http://localhost:5000, http://localhost:8080")
        if not click.confirm("Continue anyway? (Choose Yes if you're sure MLflow is running)"):
            click.echo("Setup aborted. Please ensure MLflow is running and try again.")
            return
        else:
            click.echo("Continuing with setup using the provided MLflow URI.")
    else:
        click.echo("✅ Successfully connected to MLflow!")
    
    config['mlflow_uri'] = mlflow_uri
    
    # Ask about existing prompts in MLflow
    if click.confirm("\nDo you already have prompts in MLflow that you want Core4AI to use?", default=False):
        click.echo("\nYou can provide existing prompt names to import into Core4AI.")
        click.echo("This will allow Core4AI to use these prompts for query matching.")
        
        import_method = click.prompt(
            "How would you like to import prompts?",
            type=click.Choice(['enter', 'file'], case_sensitive=False),
            default='enter'
        )
        
        if import_method.lower() == 'enter':
            # Direct input
            prompt_names = click.prompt(
                "Enter comma-separated prompt names (e.g., essay_prompt,email_prompt)",
                default=""
            )
            
            if prompt_names:
                from ..prompt_manager.registry import import_existing_prompts
                prompt_list = [name.strip() for name in prompt_names.split(",")]
                import_result = import_existing_prompts(prompt_list)
                
                click.echo(f"\n✅ Imported {import_result['imported']} prompts")
                if import_result['failed'] > 0:
                    click.echo(f"❌ Failed to import {import_result['failed']} prompts")
                    
                # Add detected prompt types to output
                if import_result.get('prompt_types'):
                    click.echo(f"🏷️  Registered prompt types: {', '.join(import_result['prompt_types'])}")
        else:
            # File input
            file_path = click.prompt(
                "Enter the path to a file containing prompt names (one per line or comma-separated)",
                type=click.Path(exists=True)
            )
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Handle both formats (comma-separated or newline-separated)
                if ',' in content:
                    prompt_list = [name.strip() for name in content.split(",")]
                else:
                    prompt_list = [name.strip() for name in content.splitlines()]
                
                prompt_list = [name for name in prompt_list if name]  # Remove empty names
                
                if prompt_list:
                    from ..prompt_manager.registry import import_existing_prompts
                    import_result = import_existing_prompts(prompt_list)
                    
                    click.echo(f"\n✅ Imported {import_result['imported']} prompts")
                    if import_result['failed'] > 0:
                        click.echo(f"❌ Failed to import {import_result['failed']} prompts")
                        
                    # Add detected prompt types to output
                    if import_result.get('prompt_types'):
                        click.echo(f"🏷️  Registered prompt types: {', '.join(import_result['prompt_types'])}")
                else:
                    click.echo("No valid prompt names found in the file.")
            except Exception as e:
                click.echo(f"Error reading file: {e}")
    
    # AI Provider
    provider_options = ['OpenAI', 'Ollama']
    provider_choice = click.prompt(
        "\nWhich AI provider would you like to use?",
        type=click.Choice(provider_options, case_sensitive=False),
        default=config.get('provider', {}).get('type', 'OpenAI').capitalize()
    )
    
    current_provider_type = provider_choice.lower()
    
    # Check if provider is changing and handle default models
    provider_changed = previous_provider and previous_provider != current_provider_type
    
    if provider_choice.lower() == 'openai':
        # Default OpenAI model
        default_model = "gpt-3.5-turbo"
        
        # If switching from another provider, show a message
        if provider_changed:
            click.echo(f"\n✅ Switching to OpenAI provider")
        
        # Initialize provider config
        provider_config = {
            'type': 'openai',
            'model': default_model  # Will be updated after user selection
        }
        
        # Check for OpenAI API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            click.echo("\n⚠️  OpenAI API key not found in environment variables.")
            click.echo("Please export your OpenAI API key as OPENAI_API_KEY.")
            click.echo("Example: export OPENAI_API_KEY='your-key-here'")
            if click.confirm("Would you like to enter your API key now? (Not recommended for security reasons)"):
                api_key = click.prompt("Enter your OpenAI API key", hide_input=True)
                provider_config['api_key'] = api_key
                click.echo("\n⚠️  Note: Your API key will be stored in the config file.")
                click.echo("For better security, consider using environment variables instead.")
            elif not click.confirm("Continue without API key?"):
                click.echo("Setup aborted. Please set the API key and try again.")
                return
        else:
            click.echo("✅ Found OpenAI API key in environment!")
        
        # Always ask for model choice
        model_options = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
        
        # If changing providers, suggest the default, otherwise use previous config
        if provider_changed:
            suggested_model = default_model
        else:
            current_model = config.get('provider', {}).get('model', default_model)
            suggested_model = current_model if current_model in model_options else default_model
        
        model = click.prompt(
            "Choose an OpenAI model",
            type=click.Choice(model_options, case_sensitive=False),
            default=suggested_model
        )
        provider_config['model'] = model
    
    elif provider_choice.lower() == 'ollama':
        # Default Ollama settings
        default_model = "llama3.2:latest"
        default_uri = "http://localhost:11434"
        
        # If switching from another provider, automatically set defaults
        if provider_changed:
            click.echo(f"\n✅ Switching to Ollama provider with default URI and model")
        
        # Ollama configuration - always ask for URI
        ollama_uri = click.prompt(
            "\nEnter your Ollama server URI",
            default=config.get('provider', {}).get('uri', default_uri)
        )
        
        # Initialize provider config with default model and user-specified URI
        provider_config = {
            'type': 'ollama',
            'uri': ollama_uri,
            'model': default_model  # Will be updated if user selects a different model
        }
        
        # Check if Ollama is running
        if not verify_ollama_running(ollama_uri):
            click.echo("\n⚠️  Warning: Ollama server not running or not accessible at this URI.")
            if click.confirm("Would you like to try starting Ollama?"):
                try:
                    # Try to start Ollama
                    subprocess.Popen(['ollama', 'serve'], 
                                     stdout=subprocess.DEVNULL, 
                                     stderr=subprocess.DEVNULL)
                    click.echo("Started Ollama server. Waiting for it to initialize...")
                    
                    # Wait for the server to start
                    for _ in range(5):  # Try for 5 seconds
                        time.sleep(1)
                        if verify_ollama_running(ollama_uri):
                            click.echo("✅ Ollama server is now running!")
                            break
                    else:
                        click.echo("⚠️  Ollama server still not responding. Continuing anyway.")
                except Exception as e:
                    click.echo(f"⚠️  Error starting Ollama: {e}")
                    if not click.confirm("Continue anyway?"):
                        click.echo("Setup aborted. Please start Ollama manually and try again.")
                        return
            elif not click.confirm("Continue anyway?"):
                click.echo("Setup aborted. Please start Ollama server and try again.")
                return
        else:
            click.echo("✅ Ollama server is running!")
        
        # Get available models
        available_models = get_ollama_models(ollama_uri)
        if available_models:
            click.echo(f"\nAvailable Ollama models: {', '.join(available_models)}")
            
            if available_models and len(available_models) > 0:
                # If changing providers, suggest the default, otherwise use previous config
                if provider_changed:
                    suggested_model = default_model if default_model in available_models else available_models[0]
                else:
                    current_model = config.get('provider', {}).get('model')
                    suggested_model = current_model if current_model in available_models else available_models[0]
                
                ollama_model = click.prompt(
                    "Choose an Ollama model",
                    type=click.Choice(available_models, case_sensitive=True),
                    default=suggested_model
                )
                provider_config['model'] = ollama_model
            else:
                ollama_model = click.prompt(
                    "Enter the Ollama model to use",
                    default=config.get('provider', {}).get('model', default_model)
                )
                provider_config['model'] = ollama_model
        else:
            ollama_model = click.prompt(
                "Enter the Ollama model to use",
                default=config.get('provider', {}).get('model', default_model)
            )
            provider_config['model'] = ollama_model
            
            # Ask if they want to pull the model
            if click.confirm(f"Would you like to pull the '{ollama_model}' model now?"):
                click.echo(f"Pulling model '{ollama_model}'... This may take a while.")
                try:
                    subprocess.run(['ollama', 'pull', ollama_model], check=True)
                    click.echo(f"✅ Successfully pulled model '{ollama_model}'!")
                except Exception as e:
                    click.echo(f"⚠️  Error pulling model: {e}")
                    if not click.confirm("Continue anyway?"):
                        click.echo("Setup aborted.")
                        return
    
    config['provider'] = provider_config
    
    # Add Analytics Configuration
    click.echo("\n📊 Analytics Configuration")
    click.echo("Core4AI can track prompt usage and performance to help you optimize your prompts.")
    
    enable_analytics = click.confirm("Would you like to enable analytics?", default=True)
    config['analytics'] = {'enabled': enable_analytics}
    
    if enable_analytics:
        import importlib
        # Set analytics storage location
        default_analytics_location = str(CONFIG_DIR / "analytics.db")
        custom_location = click.confirm("Use custom location for analytics database?", default=False)
        
        if custom_location:
            # Ask for custom location
            analytics_location = click.prompt(
                "Enter path for analytics database", 
                default=default_analytics_location
            )
            
            # Make sure parent directory exists
            try:
                parent_dir = Path(analytics_location).parent
                if not parent_dir.exists():
                    parent_dir.mkdir(parents=True)
                    click.echo(f"Created directory: {parent_dir}")
            except Exception as e:
                click.echo(f"⚠️  Error creating directory: {e}")
                click.echo("Using default location instead.")
                analytics_location = default_analytics_location
        else:
            analytics_location = default_analytics_location
        
        config['analytics']['db_path'] = analytics_location
        
        # First save the config to enable analytics before initialization
        save_config(config)

        # Then initialize analytics database
        try:
            from ..analytics.tracking import ensure_analytics_db
            # Set the database path directly in the tracking module
            from ..analytics.tracking import get_analytics_db_path
            import sys
            
            # Force reload analytics config now that we've saved it
            importlib.reload(sys.modules['core4ai.analytics.tracking'])
            
            # Now initialize the database with analytics already enabled in config
            if ensure_analytics_db():
                click.echo(f"✅ Analytics database initialized at {analytics_location}")
            else:
                click.echo(f"⚠️  Analytics database initialization failed. Analytics will be disabled.")
                config['analytics']['enabled'] = False
                # Save updated config again
                save_config(config)
        except Exception as e:
            click.echo(f"⚠️  Error initializing analytics database: {e}")
            click.echo("Analytics will be disabled.")
            config['analytics']['enabled'] = False
            # Save updated config again
            save_config(config)
    
    # Register sample prompts
    if click.confirm("\nWould you like to register the built-in sample prompts?", default=True):
        try:
            from ..prompt_manager.registry import register_sample_prompts
            result = register_sample_prompts()
            
            if result.get("status") == "success":
                click.echo(f"✅ Successfully registered {result.get('registered', 0)} sample prompts")
                if result.get("skipped", 0) > 0:
                    click.echo(f"ℹ️  Skipped {result.get('skipped', 0)} existing prompts")
            else:
                click.echo(f"⚠️  Error registering sample prompts: {result.get('error', 'Unknown error')}")
        except Exception as e:
            click.echo(f"⚠️  Error registering sample prompts: {e}")
    
    # Save the configuration
    save_config(config)
    
    click.echo("\n✅ Configuration saved successfully!")
    click.echo("\n┌──────────────────────────────────────────────────┐")
    click.echo("│               Getting Started                    │")
    click.echo("└──────────────────────────────────────────────────┘")
    click.echo("\nYou can now use Core4AI with the following commands:")
    click.echo("  core4ai register  - Register a new prompt")
    click.echo("  core4ai list      - List available prompts")
    click.echo("  core4ai chat      - Chat with AI using enhanced prompts")
    
    if enable_analytics:
        click.echo("  core4ai analytics - View prompt usage analytics")
    
    click.echo("\nFor more information, use 'core4ai --help'")