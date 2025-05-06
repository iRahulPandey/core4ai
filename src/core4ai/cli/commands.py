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
    
# Add these commands to the CLI in src/core4ai/cli/commands.py

@cli.group()
def analytics():
    """Analytics commands for prompt usage tracking.
    
    Examples:
    
    \b
    # View analytics for a specific prompt
    core4ai analytics prompt --name email_prompt
    
    \b
    # View overall usage statistics
    core4ai analytics usage --time-range 30
    
    """
    pass

@analytics.command(name="prompt")
@click.option('--name', '-n', help='Name of the prompt to analyze')
@click.option('--time-range', '-t', type=int, help='Time range in days')
@click.option('--version', '-v', type=int, help='Specific version to analyze')
@click.option('--limit', '-l', type=int, default=10, help='Maximum number of records')
@click.option('--output', '-o', help='Save results to JSON file')
def analytics_prompt(name, time_range, version, limit, output):
    """View analytics for a specific prompt or all prompts.
    
    Examples:
    
    \b
    # View analytics for email_prompt
    core4ai analytics prompt --name email_prompt
    
    \b
    # View analytics for the last 30 days
    core4ai analytics prompt --time-range 30
    
    \b
    # Export analytics data to a file
    core4ai analytics prompt --output analytics.json
    """
    from ..analytics.tracking import get_prompt_analytics
    
    try:
        result = get_prompt_analytics(name, time_range, version, limit)
        
        if result.get("status") == "success":
            # Save to file if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                click.echo(f"✅ Analytics data written to {output}")
            
            # Print summary to console
            if name:
                click.echo(f"\n📊 Analytics for prompt: {name}")
                
                # Print metrics
                metrics = result.get("metrics", [])
                if metrics:
                    click.echo("\nPrompt Metrics:")
                    click.echo(f"{'Version':<10} {'Uses':<10} {'Conf %':<10} {'Duration':<10} {'Success %':<10}")
                    click.echo("-" * 60)
                    
                    for metric in metrics:
                        version = metric.get("prompt_version", "?")
                        uses = metric.get("total_uses", 0)
                        conf = round(metric.get("avg_confidence", 0), 1)
                        duration = round(metric.get("avg_duration", 0), 2)
                        success = round(metric.get("success_rate", 0), 1)
                        
                        click.echo(f"{version:<10} {uses:<10} {conf:<10} {duration:<10}s {success:<10}")
                
                # Print version comparison
                version_comp = result.get("version_comparison", [])
                if version_comp and len(version_comp) > 1:
                    click.echo("\nVersion Comparison:")
                    for v in version_comp:
                        v_num = v.get("prompt_version", "?")
                        uses = v.get("total_uses", 0)
                        click.echo(f"Version {v_num}: {uses} uses, {round(v.get('avg_confidence', 0), 1)}% confidence")
            else:
                click.echo("\n📊 Analytics for all prompts")
                
                # Print most used prompts if available
                most_used = result.get("most_used_prompts", [])
                if most_used:
                    click.echo("\nMost Used Prompts:")
                    click.echo(f"{'Prompt':<25} {'Uses':<10} {'Conf %':<10} {'Success %':<10}")
                    click.echo("-" * 60)
                    
                    for prompt in most_used:
                        name = prompt.get("prompt_name", "?")
                        uses = prompt.get("total_uses", 0)
                        conf = round(prompt.get("avg_confidence", 0), 1)
                        success = round(prompt.get("success_rate", 0), 1)
                        
                        click.echo(f"{name:<25} {uses:<10} {conf:<10} {success:<10}")
            
            # Print provider usage if available
            provider_usage = result.get("provider_usage", [])
            if provider_usage:
                click.echo("\nProvider Performance:")
                click.echo(f"{'Provider':<15} {'Model':<15} {'Uses':<10} {'Conf %':<10} {'Duration':<10} {'Success %':<10}")
                click.echo("-" * 80)
                
                for prov in provider_usage:
                    provider = prov.get("provider", "?")
                    model = prov.get("model", "?")
                    count = prov.get("count", 0)
                    conf = round(prov.get("avg_confidence", 0), 1)
                    duration = round(prov.get("avg_duration", 0), 2)
                    success = round(prov.get("success_rate", 0), 1)
                    
                    click.echo(f"{provider:<15} {model:<15} {count:<10} {conf:<10} {duration:<10}s {success:<10}")
            
            # Show usage over time (last 7 days)
            usage_by_date = result.get("usage_by_date", [])
            if usage_by_date:
                click.echo("\nRecent Usage:")
                for date_data in usage_by_date[-7:]:  # Last 7 days
                    date = date_data.get("date", "?")
                    count = date_data.get("count", 0)
                    bar = "█" * min(int(count / 2) + 1, 30)  # Simple ASCII chart
                    click.echo(f"{date}: {count} {bar}")
            
            click.echo(f"\nShowing data for {result.get('count', 0)} usage records")
            if time_range:
                click.echo(f"Time range: Last {time_range} days")
                
            if not output:
                click.echo("\nTip: Use --output FILENAME.json to export full analytics data")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@analytics.command(name="usage")
@click.option('--time-range', '-t', type=int, default=30, help='Time range in days')
@click.option('--output', '-o', help='Save results to JSON file')
def analytics_usage(time_range, output):
    """View overall usage statistics across all prompts.
    
    Examples:
    
    \b
    # View usage stats for the last 30 days
    core4ai analytics usage
    
    \b
    # View usage stats for the last 90 days
    core4ai analytics usage --time-range 90
    
    \b
    # Export usage data to a file
    core4ai analytics usage --output usage.json
    """
    from ..analytics.tracking import get_usage_stats
    
    try:
        result = get_usage_stats(time_range)
        
        if result.get("status") == "success":
            # Save to file if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                click.echo(f"✅ Usage data written to {output}")
            
            # Print summary to console
            if time_range:
                click.echo(f"\n📊 Usage Statistics (Last {time_range} days)")
            else:
                click.echo("\n📊 Usage Statistics (All time)")
                
            click.echo(f"Total Usages: {result.get('total_count', 0)}")
            
            # Print usage by prompt
            usage_by_prompt = result.get("usage_by_prompt", [])
            if usage_by_prompt:
                click.echo("\nUsage by Prompt Type:")
                click.echo(f"{'Prompt':<25} {'Count':<10} {'Conf %':<10} {'Duration':<10} {'Success %':<10}")
                click.echo("-" * 70)
                
                for prompt in usage_by_prompt[:10]:  # Top 10
                    name = prompt.get("prompt_name", "?")
                    count = prompt.get("count", 0)
                    conf = round(prompt.get("avg_confidence", 0), 1)
                    duration = round(prompt.get("avg_duration", 0), 2)
                    success = round(prompt.get("success_rate", 0), 1)
                    
                    click.echo(f"{name:<25} {count:<10} {conf:<10} {duration:<10}s {success:<10}")
            
            # Print provider stats
            provider_stats = result.get("provider_stats", [])
            if provider_stats:
                click.echo("\nProvider Statistics:")
                click.echo(f"{'Provider':<15} {'Model':<15} {'Count':<8} {'Fallback %':<10} {'Success %':<10}")
                click.echo("-" * 70)
                
                for prov in provider_stats:
                    provider = prov.get("provider", "?")
                    model = prov.get("model", "?")
                    count = prov.get("count", 0)
                    fallback = round(prov.get("fallback_rate", 0), 1)
                    success = round(prov.get("success_rate", 0), 1)
                    
                    click.echo(f"{provider:<15} {model:<15} {count:<8} {fallback:<10} {success:<10}")
            
            # Print content type stats
            content_stats = result.get("content_stats", [])
            if content_stats:
                click.echo("\nContent Type Statistics:")
                click.echo(f"{'Content Type':<20} {'Count':<8} {'Success %':<10}")
                click.echo("-" * 50)
                
                for content in content_stats[:10]:  # Top 10
                    c_type = content.get("content_type", "?")
                    count = content.get("count", 0)
                    success = round(content.get("success_rate", 0), 1)
                    
                    click.echo(f"{c_type:<20} {count:<8} {success:<10}")
            
            # Print usage by day
            usage_by_day = result.get("usage_by_day", [])
            if usage_by_day:
                click.echo("\nUsage Trend:")
                
                # Show last 7 days
                for day_data in usage_by_day[-7:]:
                    day = day_data.get("date", "?")
                    count = day_data.get("count", 0)
                    bar = "█" * min(int(count / 2) + 1, 30)  # Simple ASCII chart
                    click.echo(f"{day}: {count} {bar}")
                    
            if not output:
                click.echo("\nTip: Use --output FILENAME.json to export full usage data")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@analytics.command(name="clear")
@click.option('--name', '-n', help='Name of the prompt to clear data for (omit to clear all)')
@click.confirmation_option(prompt='Are you sure you want to clear analytics data?')
def analytics_clear(name):
    """Clear analytics data.
    
    Examples:
    
    \b
    # Clear data for a specific prompt
    core4ai analytics clear --name email_prompt
    
    \b
    # Clear all analytics data
    core4ai analytics clear
    """
    from ..analytics.tracking import clear_analytics
    
    try:
        result = clear_analytics(name)
        
        if result.get("status") == "success":
            click.echo(f"✅ {result.get('message', 'Analytics data cleared')}")
            click.echo(f"Rows affected: {result.get('rows_affected', 0)}")
        else:
            click.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

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

@analytics.command(name="dashboard")
@click.option('--output-dir', '-o', help='Directory to save the dashboard')
@click.option('--filename', '-f', help='Filename for the dashboard')
@click.option('--time-range', '-t', type=int, default=30, help='Time range in days')
@click.option('--open', is_flag=True, help='Open the dashboard in browser after generation')
def analytics_dashboard(output_dir, filename, time_range, open):
    """Generate an HTML dashboard with analytics data.
    
    Examples:
    
    \b
    # Generate a dashboard in current directory
    core4ai analytics dashboard
    
    \b
    # Generate dashboard with custom filename and location
    core4ai analytics dashboard --output-dir ./reports --filename my_dashboard.html
    
    \b
    # Generate dashboard for the last 90 days and open in browser
    core4ai analytics dashboard --time-range 90 --open
    """
    from ..analytics.tracking import get_prompt_analytics, get_usage_stats
    from ..utils.dashboard import generate_dashboard
    
    try:
        # Get analytics data
        analytics_data = get_prompt_analytics(time_range=time_range)
        usage_data = get_usage_stats(time_range=time_range)
        
        # Check if analytics is enabled
        if analytics_data.get("status") == "error" or usage_data.get("status") == "error":
            if analytics_data.get("error", "").startswith("Analytics is disabled"):
                click.echo("❌ Analytics is disabled. Enable it in the configuration.")
                return
            
            # Some other error occurred
            error_msg = analytics_data.get("error") or usage_data.get("error")
            click.echo(f"❌ Error retrieving analytics data: {error_msg}")
            return
        
        # Generate the dashboard
        dashboard_path = generate_dashboard(
            analytics_data, 
            usage_data, 
            output_dir=output_dir,
            filename=filename
        )
        
        click.echo(f"✅ Dashboard generated at: {dashboard_path}")
        
        # Open the dashboard in browser if requested
        if open:
            try:
                import webbrowser
                click.echo("Opening dashboard in browser...")
                webbrowser.open(f"file://{dashboard_path}")
            except Exception as e:
                click.echo(f"⚠️ Could not open browser: {e}")
        
    except Exception as e:
        click.echo(f"❌ Error generating dashboard: {e}")

@cli.command()
@click.argument('query', required=False)
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output and processing information')
@click.option('--simple', '-s', is_flag=True, help='Show only the response for one-off queries (no details)')
def chat(query, verbose, simple):
    """Chat with AI using enhanced prompts.
    
    If a query is provided, processes a single query and returns the result.
    If no query is provided, starts an interactive chat session.
    
    Examples:
    
    \b
    # Simple one-off query
    core4ai chat "Write an essay about climate change"
    
    \b
    # Start interactive chat session (simple by default)
    core4ai chat
    
    \b
    # Start interactive chat session with detailed processing info
    core4ai chat --verbose
    
    \b
    # Show only the response for one-off query, no details
    core4ai chat --simple "Write an email to my boss"
    """
    import time  # Add the time import
    import logging

    # Always save the current logging level regardless of mode
    original_level = logging.getLogger().level
    
    # If simple mode, suppress all logging temporarily
    if simple:
        # Set logging to error level only to suppress info logging
        logging.getLogger().setLevel(logging.ERROR)

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
    
    # One-off mode if query is provided
    if query:
        # Process the query (existing behavior)
        if verbose and not simple:
            click.echo(f"Processing query: {query}")
            click.echo(f"Using provider: {provider_config['type']}")
            click.echo(f"Using model: {provider_config.get('model', 'default')}")
        
        try:
            # Process the query
            result = asyncio.run(process_query(query, provider_config, verbose and not simple))
            
            # Display results
            if simple:
                # Simple output - just the response
                # Restore logging level before output
                logging.getLogger().setLevel(original_level)
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
        except Exception as e:
            # Restore logging level before showing error
            if simple:
                logging.getLogger().setLevel(original_level)
            click.echo(f"Error processing query: {e}")
        
        # If exiting after a simple one-off query, make sure to restore logging
        if simple:
            logging.getLogger().setLevel(original_level)
            
        return

    # If simple mode was set, restore logging level for interactive mode
    if simple:
        logging.getLogger().setLevel(original_level)

    # Set up logging for interactive mode based on verbose flag
    interactive_log_level = original_level
    if not verbose:
        # If not verbose, suppress logging in interactive mode
        logging.getLogger().setLevel(logging.ERROR)

    # Interactive mode if no query provided
    # Always display connection info in both modes
    click.echo("\n🧠 Core4AI Interactive Mode")
    click.echo(f"Connected to MLflow at: {mlflow_uri}")
    click.echo(f"Using {provider_config.get('type').upper()} with model: {provider_config.get('model', 'default')}")
    click.echo("\nType your questions and press Enter. Type /bye to exit.")
    click.echo("=" * 70)
    
    # Track conversation history
    conversation_history = []
    
    try:
        while True:
            # Always use emojis for the chat interface
            try:
                user_query = click.prompt("\n🧑", prompt_suffix="")
            except click.exceptions.Abort:
                # Use consistent goodbye message in both modes
                click.echo("\nGoodbye! Thank you for using Core4AI Assistant.")
                break
            
            # Check for exit command
            if user_query.lower() in ['/bye', '/exit', '/quit']:
                # Use consistent goodbye message in both modes
                click.echo("\nGoodbye! Thank you for using Core4AI Assistant.")
                break
            
            # Process the query
            try:
                # Handle progress indication differently based on verbose mode
                if verbose:
                    # In verbose mode, use the progress bar
                    with click.progressbar(length=100, label='Thinking...', show_eta=False) as bar:
                        for i in range(100):
                            time.sleep(0.1)  # Small delay for visual effect
                            bar.update(1)
                        
                        # Process query
                        result = asyncio.run(process_query(user_query, provider_config, True))
                else:
                    # In simple mode, just show a minimal indicator or spinner
                    with click.progressbar(length=100, label='Thinking...', show_eta=False) as bar:
                        for i in range(100):
                            time.sleep(0.1)  # Small delay for visual effect
                            bar.update(1)
                        
                        # Process query
                        result = asyncio.run(process_query(user_query, provider_config, False))
                
                # Display results with emoji but level of detail based on verbose flag
                click.echo("\n🤖 ", nl=False)  # Always keep the robot emoji
                
                if verbose:
                    # Show match details in verbose mode
                    prompt_match = result.get("prompt_match", {})
                    match_status = prompt_match.get("status", "unknown")
                    
                    if match_status == "matched":
                        click.echo(f"[Matched to: {prompt_match.get('prompt_name')} with {prompt_match.get('confidence')}% confidence]")
                        if result.get("content_type"):
                            click.echo(f"[Content Type: {result['content_type']}]")
                            
                        if result.get("enhanced", False):
                            click.echo(f"[Enhanced Query: {result.get('enhanced_query', 'N/A')}]")
                            
                        click.echo("\n")
                
                # Display the response
                click.echo(result.get('response', 'No response received.'))
                
                # Add to conversation history
                conversation_history.append({
                    "query": user_query,
                    "response": result.get('response', 'No response received.')
                })
                
            except Exception as e:
                # Keep emoji for error messages too
                click.echo(f"\n🤖 Error processing query: {e}")
    finally:
        # Restore logging level when exiting interactive mode
        if not verbose:
            logging.getLogger().setLevel(interactive_log_level)

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