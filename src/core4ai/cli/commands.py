import click
import json
import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Internal imports
from ..config.config import load_config, get_mlflow_uri, get_provider_config
from ..prompt_manager.registry import (
    register_prompt, register_from_file, list_prompts as registry_list_prompts,
    register_sample_prompts, update_prompt, get_prompt_details
)
from ..client.client import process_query
from .setup import setup_wizard

# Set up logging
logger = logging.getLogger("core4ai.cli")

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Core4AI: Contextual Optimization and Refinement Engine for AI.
    
    This CLI tool helps you manage prompts and interact with AI providers.
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@cli.command()
def setup():
    """Run the interactive setup wizard."""
    setup_wizard()

@cli.command()
def serve():
    """Start the Core4AI server (coming soon)."""
    click.echo("The server mode is not yet implemented in this version.")
    click.echo("Core4AI currently operates in direct mode without a separate server.")
    click.echo("You can use 'core4ai chat' to interact with AI directly.")

@cli.command()
@click.argument('prompt', required=False)
@click.option('--file', '-f', help='Register prompts from a JSON file')
@click.option('--name', '-n', help='Name for the prompt')
@click.option('--message', '-m', default='Registered via CLI', help='Commit message')
@click.option('--tags', '-t', help='Tags as JSON string')
@click.option('--samples', is_flag=True, help='Register sample prompts')
@click.option('--no-production', is_flag=True, help="Don't set as production alias")
def register(prompt, file, name, message, tags, samples, no_production):
    """Register a new prompt or prompts from a file."""
    # Check if we're installing samples
    if samples:
        click.echo("Registering sample prompts...")
        result = register_sample_prompts()
        click.echo(json.dumps(result, indent=2))
        return
    
    # Otherwise we need either a prompt or a file
    if not prompt and not file:
        click.echo("Error: Please provide either a prompt, a file, or use --samples flag.")
        sys.exit(1)
    
    # Check for MLflow URI
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        click.echo("Error: MLflow URI not configured. Run 'core4ai setup' first.")
        sys.exit(1)
    
    if file:
        # Register from file
        if not Path(file).exists():
            click.echo(f"Error: File '{file}' not found.")
            sys.exit(1)
        
        result = register_from_file(file, set_as_production=not no_production)
        click.echo(json.dumps(result, indent=2))
    
    else:
        # Register direct prompt
        if not name:
            name = click.prompt("Enter a name for this prompt")
        
        parsed_tags = {}
        if tags:
            try:
                parsed_tags = json.loads(tags)
            except json.JSONDecodeError:
                click.echo("Error: Tags must be a valid JSON string.")
                sys.exit(1)
        
        result = register_prompt(
            name=name,
            template=prompt,
            commit_message=message,
            tags=parsed_tags,
            set_as_production=not no_production
        )
        
        click.echo(json.dumps(result, indent=2))

@cli.command()
@click.option('--details', '-d', is_flag=True, help='Show detailed information')
@click.option('--name', '-n', help='Get details for a specific prompt')
def list(details, name):
    """List all available prompts."""
    # Check for MLflow URI
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        click.echo("Error: MLflow URI not configured. Run 'core4ai setup' first.")
        sys.exit(1)
    
    if name:
        # Get details for a specific prompt
        result = get_prompt_details(name)
        if result.get("status") == "success":
            click.echo(f"Prompt: {result['name']}")
            click.echo(f"Latest Version: {result['latest_version']}")
            
            if result.get('production_version'):
                click.echo(f"Production Version: {result['production_version']}")
            
            if result.get('archived_versions'):
                click.echo(f"Archived Versions: {', '.join(map(str, result['archived_versions']))}")
            
            click.echo(f"Variables: {', '.join(result['variables'])}")
            
            if result.get('tags'):
                click.echo(f"Tags: {json.dumps(result['tags'])}")
            
            if details:
                click.echo("\nTemplate:")
                click.echo("------------------------------")
                click.echo(result['latest_template'])
                click.echo("------------------------------")
        else:
            click.echo(f"Error: {result.get('error', 'Unknown error')}")
    else:
        # List all prompts
        result = registry_list_prompts()
        if result.get("status") == "success":
            prompts = result.get("prompts", [])
            if prompts:
                if details:
                    # Detailed output as JSON
                    click.echo(json.dumps(prompts, indent=2))
                else:
                    # Simple table output
                    click.echo(f"Found {len(prompts)} prompts:")
                    
                    # Headers
                    headers = ["Name", "Type", "Variables", "Version"]
                    if details:
                        headers.extend(["Production", "Archived"])
                    
                    # Format and print
                    row_format = "{:<25} {:<15} {:<30} {:<10}"
                    if details:
                        row_format += " {:<10} {:<10}"
                    
                    click.echo(row_format.format(*headers))
                    click.echo("-" * 80)
                    
                    for prompt in prompts:
                        vars_str = ", ".join(prompt.get("variables", [])[:3])
                        if len(prompt.get("variables", [])) > 3:
                            vars_str += "..."
                        
                        row = [
                            prompt["name"], 
                            prompt["type"], 
                            vars_str, 
                            str(prompt.get("latest_version", "N/A"))
                        ]
                        
                        if details:
                            row.extend([
                                str(prompt.get("production_version", "N/A")),
                                str(prompt.get("archived_version", "N/A"))
                            ])
                        
                        click.echo(row_format.format(*row))
            else:
                click.echo("No prompts found. Use 'core4ai register --samples' to register sample prompts.")
        else:
            click.echo(f"Error listing prompts: {result.get('error', 'Unknown error')}")

@cli.command()
@click.argument('query')
@click.option('--verbose', '-v', is_flag=True, help='Show verbose output')
@click.option('--simple', '-s', is_flag=True, help='Show only the response (no enhancement details)')
def chat(query, verbose, simple):
    """Chat with AI using enhanced prompts.
    
    Uses the provider configured during 'core4ai setup'.
    To change providers, run 'core4ai setup' again.
    """
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        click.echo("Error: MLflow URI not configured. Run 'core4ai setup' first.")
        sys.exit(1)
    
    # Get provider config
    provider_config = get_provider_config()
    if not provider_config or not provider_config.get('type'):
        click.echo("Error: AI provider not configured. Run 'core4ai setup' first.")
        sys.exit(1)
    
    # Ensure Ollama has a URI if that's the configured provider
    if provider_config.get('type') == 'ollama' and not provider_config.get('uri'):
        provider_config['uri'] = 'http://localhost:11434'
    
    if verbose:
        click.echo(f"Processing query: {query}")
        click.echo(f"Using provider: {provider_config['type']}")
        click.echo(f"Using model: {provider_config.get('model', 'default')}")
    
    # Process the query
    result = asyncio.run(process_query(query, provider_config, verbose))
    
    # Display results
    if simple:
        # Simple output - just the response
        click.echo(result.get('response', 'No response received.'))
    else:
        # Detailed traceability output like promptlab
        prompt_match = result.get("prompt_match", {})
        match_status = prompt_match.get("status", "unknown")
        
        click.echo("\n=== Core4AI Results ===\n")
        click.echo(f"Original Query: {result['original_query']}")
        
        if match_status == "matched":
            click.echo(f"\nMatched to: {prompt_match.get('prompt_name')}")
            click.echo(f"Confidence: {prompt_match.get('confidence')}%")
            if verbose and prompt_match.get('reasoning'):
                click.echo(f"Reasoning: {prompt_match.get('reasoning')}")
        elif match_status == "no_match":
            click.echo("\nNo matching prompt template found.")
            if verbose and prompt_match.get('reasoning'):
                click.echo(f"Reason: {prompt_match.get('reasoning')}")
        elif match_status == "no_prompts_available":
            click.echo("\nNo prompts available in MLflow registry.")
        
        if result.get("content_type"):
            click.echo(f"Content Type: {result['content_type']}")
        
        # Show the enhanced query details if enhancement was performed
        if result.get("enhanced", False):
            # Always show the initial enhanced query
            if result.get("initial_enhanced_query"):
                click.echo("\nInitial Enhanced Query:")
                click.echo("-" * 80)
                click.echo(result['initial_enhanced_query'])
                click.echo("-" * 80)
                
                if result.get("validation_issues"):
                    click.echo("\nValidation Issues Detected:")
                    for issue in result["validation_issues"]:
                        click.echo(f"- {issue}")
                
                # Show the adjusted query if it's different
                if result.get("enhanced_query") and result.get("enhanced_query") != result.get("initial_enhanced_query"):
                    click.echo("\nAdjusted Query:")
                    click.echo("-" * 80)
                    click.echo(result['enhanced_query'])
                    click.echo("-" * 80)
            elif result.get("enhanced_query"):
                click.echo("\nEnhanced Query:")
                click.echo("-" * 80)
                click.echo(result['enhanced_query'])
                click.echo("-" * 80)
        else:
            click.echo("\nUsing original query (no enhancement applied)")
        
        click.echo("\nResponse:")
        click.echo("=" * 80)
        click.echo(result.get('response', 'No response received.'))
        click.echo("=" * 80)

@cli.command()
def version():
    """Show Core4AI version information."""
    from .. import __version__
    
    click.echo(f"Core4AI version: {__version__}")
    
    # Show configuration
    config = load_config()
    mlflow_uri = config.get('mlflow_uri', 'Not configured')
    provider = config.get('provider', {}).get('type', 'Not configured')
    model = config.get('provider', {}).get('model', 'default')
    
    click.echo(f"MLflow URI: {mlflow_uri}")
    click.echo(f"Provider: {provider}")
    click.echo(f"Model: {model}")
    
    # Show system information
    import platform
    import sys
    
    click.echo(f"Python version: {platform.python_version()}")
    click.echo(f"System: {platform.system()} {platform.release()}")

if __name__ == "__main__":
    cli()