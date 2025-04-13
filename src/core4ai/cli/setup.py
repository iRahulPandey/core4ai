# core4ai/cli/setup.py
import os
import click
import requests
import logging
import subprocess
import time
from pathlib import Path
from ..config.config import load_config, save_config, ensure_config_dir
from ..client.client import verify_ollama_running, get_ollama_models

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

def setup_wizard():
    """Interactive setup wizard for core4ai."""
    click.echo("┌──────────────────────────────────────────────────────┐")
    click.echo("│             Core4AI Setup Wizard                     │")
    click.echo("│ Contextual Optimization and Refinement Engine for AI │")
    click.echo("└──────────────────────────────────────────────────────┘")
    
    click.echo("\nThis wizard will help you configure Core4AI for your environment.")
    
    # Initialize config
    config = load_config()
    
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
    
    # AI Provider
    provider_options = ['OpenAI', 'Ollama']
    provider_choice = click.prompt(
        "\nWhich AI provider would you like to use?",
        type=click.Choice(provider_options, case_sensitive=False),
        default=config.get('provider', {}).get('type', 'OpenAI').capitalize()
    )
    
    provider_config = {'type': provider_choice.lower()}
    
    if provider_choice.lower() == 'openai':
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
        
        # Let user choose model if they want
        model_options = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
        if click.confirm("\nWould you like to specify an OpenAI model?", default=False):
            model = click.prompt(
                "Choose a model",
                type=click.Choice(model_options, case_sensitive=False),
                default=config.get('provider', {}).get('model', 'gpt-3.5-turbo')
            )
            provider_config['model'] = model
        else:
            provider_config['model'] = config.get('provider', {}).get('model', 'gpt-3.5-turbo')
    
    elif provider_choice.lower() == 'ollama':
        # Ollama configuration
        ollama_uri = click.prompt(
            "\nEnter your Ollama server URI",
            default=config.get('provider', {}).get('uri', 'http://localhost:11434')
        )
        
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
                default_model = config.get('provider', {}).get('model', available_models[0])
                ollama_model = click.prompt(
                    "Choose an Ollama model",
                    type=click.Choice(available_models, case_sensitive=True),
                    default=default_model if default_model in available_models else available_models[0]
                )
            else:
                ollama_model = click.prompt(
                    "Enter the Ollama model to use",
                    default=config.get('provider', {}).get('model', 'llama2')
                )
        else:
            ollama_model = click.prompt(
                "Enter the Ollama model to use",
                default=config.get('provider', {}).get('model', 'llama2')
            )
            
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
        
        provider_config['uri'] = ollama_uri
        provider_config['model'] = ollama_model
    
    config['provider'] = provider_config
    
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
    click.echo("\nFor more information, use 'core4ai --help'")