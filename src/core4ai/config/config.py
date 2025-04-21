import os
import yaml
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("core4ai.config")

CONFIG_DIR = Path.home() / ".core4ai"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

def ensure_config_dir():
    """Ensure the configuration directory exists."""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True)
        logger.info(f"Created configuration directory at {CONFIG_DIR}")

def load_config():
    """Load configuration from file."""
    if not CONFIG_FILE.exists():
        logger.info(f"No configuration file found at {CONFIG_FILE}")
        return {}

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded configuration: {config}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def save_config(config):
    """Save configuration to file."""
    ensure_config_dir()
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")

def get_mlflow_uri():
    """Get the MLflow URI from config or environment."""
    # Environment variable should take precedence
    if mlflow_uri_env := os.environ.get('MLFLOW_TRACKING_URI'):
        return mlflow_uri_env
    
    # Fall back to config
    config = load_config()
    return config.get('mlflow_uri')

def get_analytics_config():
    """Get the analytics configuration."""
    config = load_config()
    analytics_config = config.get('analytics', {})
    
    # Set default values if not configured
    if 'enabled' not in analytics_config:
        analytics_config['enabled'] = False
        
    if 'db_path' not in analytics_config:
        analytics_config['db_path'] = str(CONFIG_DIR / "analytics.db")
        
    return analytics_config

def set_analytics_config(enabled: bool, db_path: Optional[str] = None):
    """Set the analytics configuration.
    
    Args:
        enabled: Whether analytics is enabled
        db_path: Path to the analytics database file
    """
    config = load_config()
    
    # Initialize analytics config if it doesn't exist
    if 'analytics' not in config:
        config['analytics'] = {}
        
    # Update values
    config['analytics']['enabled'] = enabled
    
    if db_path:
        config['analytics']['db_path'] = db_path
    elif 'db_path' not in config['analytics']:
        config['analytics']['db_path'] = str(CONFIG_DIR / "analytics.db")
        
    # Save updated config
    save_config(config)
    
    # If analytics is being enabled, ensure the database is initialized
    if enabled:
        from ..analytics.tracking import ensure_analytics_db
        ensure_analytics_db()

def get_provider_config():
    """Get the AI provider configuration."""
    config = load_config()
    provider = config.get('provider', {})
    
    provider_type = provider.get('type')
    
    if provider_type == 'openai':
        # Environment variable should take precedence
        api_key = os.environ.get('OPENAI_API_KEY') or provider.get('api_key')
        return {
            'type': 'openai', 
            'api_key': api_key,
            'model': provider.get('model', 'gpt-3.5-turbo')
        }
    
    elif provider_type == 'ollama':
        return {
            'type': 'ollama',
            'uri': provider.get('uri', 'http://localhost:11434'),
            'model': provider.get('model', 'llama2')
        }
    
    return {'type': None}