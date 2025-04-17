# Simplified CLI commands for src/core4ai/cli/commands.py

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
    register_prompt, register_from_file, register_from_markdown, list_prompts as registry_list_prompts,
    register_sample_prompts, update_prompt, get_prompt_details, create_prompt_template
)
from ..engine.processor import process_query
from ..providers.utilities import verify_ollama_running, get_ollama_models
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
    """Run the interactive setup wizard.
    
    This wizard helps you configure Core4AI with MLflow and your preferred AI provider.
    """
    setup_wizard()

@cli.command()
@click.argument('prompt', required=False)
@click.option('--file', '-f', help='Register prompts from a JSON file')
@click.option('--markdown', '-m', help='Register a prompt from a markdown file')
@click.option('--name', '-n', help='Name for the prompt')
@click.option('--dir', '-d', help='Directory with markdown prompt files to register')
@click.option('--samples', is_flag=True, help='Register built-in sample prompts')
@click.option('--only-new', is_flag=True, help='Only register prompts that don\'t exist yet')
@click.option('--no-production', is_flag=True, help="Don't set as production version")
@click.option('--create', '-c', is_flag=True, help='Create a new prompt template file first')
def register(prompt, file, markdown, name, dir, samples, only_new, no_production, create):
    """Register prompts from various sources.
    
    Examples:
    
    \b
    # Register a single prompt template directly
    core4ai register --name "email_prompt" "Write a {{ formality }} email..."
    
    \b
    # Register a prompt from a markdown file
    core4ai register --markdown ./my_prompts/email_prompt.md
    
    \b
    # Register all prompts from a directory
    core4ai register --dir ./my_prompts
    
    \b
    # Register built-in sample prompts
    core4ai register --samples
    
    \b
    # Create and edit a new prompt template first
    core4ai register --create email
    """
    # MLflow connection check
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        click.echo("❌ Error: MLflow URI not configured. Run 'core4ai setup' first.")
        sys.exit(1)
    
    # Option: Create a new template file first
    if create:
        prompt_name = name or prompt  # Use either --name or the argument
        
        if not prompt_name:
            prompt_name = click.prompt("Enter a name for the new prompt (e.g., email, blog, analysis)")
            
        # Create the template
        output_dir = dir if dir else None
        result = create_prompt_template(
            prompt_name=prompt_name,
            output_dir=Path(output_dir) if output_dir else None
        )
        
        if result["status"] == "success":
            filepath = result["file_path"]
            click.echo(f"✅ Created prompt template at: {filepath}")
            
            # Ask if they want to edit it
            if click.confirm("Would you like to edit this template now?", default=True):
                # Try to open in default editor
                try:
                    import subprocess
                    if sys.platform == 'win32':
                        os.startfile(filepath)
                    elif sys.platform == 'darwin':  # macOS
                        subprocess.call(['open', filepath])
                    else:  # Linux or other Unix
                        subprocess.call(['xdg-open', filepath])
                    
                    # Wait for user to finish editing
                    click.echo("Edit the template and save it, then press Enter to continue...")
                    input()
                except Exception as e:
                    click.echo(f"Could not open editor. Please edit the file manually: {filepath}")
                    click.echo(f"Press Enter when done...")
                    input()
            
            # Ask if they want to register it now
            if click.confirm("Would you like to register this prompt now?", default=True):
                # Register the newly created file
                register_result = register_from_markdown(filepath, set_as_production=not no_production)
                
                if register_result["status"] == "success":
                    click.echo(f"✅ Successfully registered: {register_result.get('name')}")
                else:
                    click.echo(f"❌ Error: {register_result.get('error', 'Unknown error')}")
            else:
                click.echo(f"You can register it later with: core4ai register --markdown {filepath}")
        else:
            click.echo(f"❌ Error creating template: {result.get('error', 'Unknown error')}")
        
        return
    
    # Option: Register sample prompts
    if samples:
        click.echo("Registering sample prompts...")
        result = register_sample_prompts(
            all_prompts=not only_new, 
            custom_dir=dir,
            non_existing_only=only_new
        )
        
        # Show results summary
        if result["status"] == "success":
            if result['registered'] > 0:
                click.echo(f"✅ Successfully registered {result['registered']} prompts")
            else:
                click.echo("ℹ️ No new prompts were registered")
                
            if result.get("skipped", 0) > 0:
                click.echo(f"↩ Skipped {result['skipped']} existing prompts")
                
            if click.confirm("View details?", default=False):
                click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return
    
    # Option: Register from directory
    if dir and not samples:
        click.echo(f"Registering prompts from directory: {dir}")
        result = register_sample_prompts(
            all_prompts=not only_new,
            custom_dir=dir,
            non_existing_only=only_new
        )
        
        # Show results summary
        if result["status"] == "success":
            click.echo(f"✅ Successfully registered {result['registered']} prompts")
            if result.get("skipped", 0) > 0:
                click.echo(f"↩ Skipped {result['skipped']} existing prompts")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return
    
    # Option: Register from markdown file
    if markdown:
        if not Path(markdown).exists():
            click.echo(f"❌ Error: File '{markdown}' not found.")
            sys.exit(1)
        
        click.echo(f"Registering prompt from markdown file: {markdown}")
        result = register_from_markdown(markdown, set_as_production=not no_production)
        
        if result["status"] == "success":
            click.echo(f"✅ Successfully registered: {result.get('name')}")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return
    
    # Option: Register from JSON file
    if file:
        if not Path(file).exists():
            click.echo(f"❌ Error: File '{file}' not found.")
            sys.exit(1)
        
        click.echo(f"Registering prompts from JSON file: {file}")
        result = register_from_file(file, set_as_production=not no_production)
        
        if result["status"] == "success":
            click.echo(f"✅ Successfully registered {result.get('count', 0)} prompts")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return
    
    # Option: Register from direct input
    if prompt:
        # Handle the prompt name
        if not name:
            name = click.prompt("Enter a name for this prompt")
            
            # Check naming convention
            if not name.endswith("_prompt"):
                if click.confirm(f"Add '_prompt' suffix to name? ({name}_prompt)"):
                    name = f"{name}_prompt"
        
        # Extract type from name
        if "_" in name:
            prompt_type = name.split("_")[0]
            
            # Add to prompt type registry
            from ..prompt_manager.prompt_types import add_prompt_type
            add_prompt_type(prompt_type)
        
        # Register the prompt
        result = register_prompt(
            name=name,
            template=prompt,
            commit_message="Registered via CLI",
            tags={"type": name.split("_")[0]} if "_" in name else {},
            set_as_production=not no_production
        )
        
        if result["status"] == "success":
            click.echo(f"✅ Successfully registered: {name}")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return
    
    # No valid option provided
    click.echo("❌ Error: Please provide a prompt or specify a source (--file, --markdown, --dir, --samples).")
    click.echo("\nRun 'core4ai register --help' to see usage examples.")
    sys.exit(1)

@cli.command()
@click.option('--details', '-d', is_flag=True, help='Show detailed information')
@click.option('--name', '-n', help='Get details for a specific prompt')
def list(details, name):
    """List available prompts.
    
    Examples:
    
    \b
    # List all prompts
    core4ai list
    
    \b
    # Show detailed information
    core4ai list --details
    
    \b
    # Get details for a specific prompt
    core4ai list --name essay_prompt
    """
    # Check for MLflow URI
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        click.echo("❌ Error: MLflow URI not configured. Run 'core4ai setup' first.")
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
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
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
                    
                    # Format and print
                    row_format = "{:<25} {:<15} {:<30} {:<10}"
                    
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
                        
                        click.echo(row_format.format(*row))
            else:
                click.echo("No prompts found. Use 'core4ai register --samples' to register sample prompts.")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")

@cli.command()
def list_types():
    """List all registered prompt types.
    
    Examples:
    
    \b
    # List all prompt types
    core4ai list-types
    """
    from ..prompt_manager.prompt_types import get_prompt_types
    
    prompt_types = get_prompt_types()
    
    if prompt_types:
        click.echo(f"Found {len(prompt_types)} registered prompt types:")
        for prompt_type in sorted(prompt_types):
            click.echo(f"- {prompt_type}")
    else:
        click.echo("No prompt types registered yet.")
        click.echo("\nRegister sample prompts to add default types:")
        click.echo("  core4ai register --samples")

@cli.command()
@click.argument('query')
@click.option('--verbose', '-v', is_flag=True, help='Show verbose output')
@click.option('--simple', '-s', is_flag=True, help='Show only the response (no enhancement details)')
def chat(query, verbose, simple):
    """Chat with AI using enhanced prompts.
    
    Examples:
    
    \b
    # Simple query
    core4ai chat "Write an essay about climate change"
    
    \b
    # Show only the response, no details
    core4ai chat --simple "Write an email to my boss"
    
    \b
    # Show detailed information about prompt selection
    core4ai chat --verbose "Create a technical explanation of quantum computing"
    """
    mlflow_uri = get_mlflow_uri()
    if not mlflow_uri:
        click.echo("❌ Error: MLflow URI not configured. Run 'core4ai setup' first.")
        sys.exit(1)
    
    # Get provider config
    provider_config = get_provider_config()
    if not provider_config or not provider_config.get('type'):
        click.echo("❌ Error: AI provider not configured. Run 'core4ai setup' first.")
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
        # Detailed traceability output
        prompt_match = result.get("prompt_match", {})
        match_status = prompt_match.get("status", "unknown")
        
        click.echo("\n=== Core4AI Results ===\n")
        click.echo(f"Original Query: {query}")
        
        if match_status == "matched":
            click.echo(f"\nMatched to: {prompt_match.get('prompt_name')}")
            click.echo(f"Confidence: {prompt_match.get('confidence')}%")
            if verbose and prompt_match.get('reasoning'):
                click.echo(f"Reasoning: {prompt_match.get('reasoning')}")
        elif match_status == "no_match":
            click.echo("\nNo matching prompt template found.")
        elif match_status == "no_prompts_available":
            click.echo("\nNo prompts available. Register some prompts first.")
        
        if result.get("content_type"):
            click.echo(f"Content Type: {result['content_type']}")
        
        # Show the enhanced query if available
        if result.get("enhanced", False) and result.get("enhanced_query"):
            click.echo("\nEnhanced Query:")
            click.echo("-" * 80)
            click.echo(result['enhanced_query'])
            click.echo("-" * 80)
        
        click.echo("\nResponse:")
        click.echo("=" * 80)
        click.echo(result.get('response', 'No response received.'))
        click.echo("=" * 80)

@cli.command()
def version():
    """Show Core4AI version information.
    
    Examples:
    
    \b
    # Show version information
    core4ai version
    """
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
    
    # Show registered prompt types
    from ..prompt_manager.prompt_types import get_prompt_types
    prompt_types = get_prompt_types()
    if prompt_types:
        click.echo(f"Registered prompt types: {len(prompt_types)}")
    
    # Show system information
    import platform
    import sys
    
    click.echo(f"Python version: {platform.python_version()}")
    click.echo(f"System: {platform.system()} {platform.release()}")

if __name__ == "__main__":
    cli()