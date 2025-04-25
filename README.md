![alt text](logo.png)

# Core4AI: Contextual Optimization and Refinement Engine for AI

[![PyPI Downloads](https://static.pepy.tech/badge/core4ai)](https://pepy.tech/projects/core4ai)

Core4AI is an intelligent system that transforms basic user queries into optimized prompts for AI systems using MLflow Prompt Registry. It dynamically matches user requests to the most appropriate prompt template and applies it with extracted parameters.

## Architecture

Core4AI's architecture is designed for seamless integration with MLflow while providing flexibility in AI provider selection:

![alt text](architecture.png)

This integration allows Core4AI to leverage MLflow's tracking capabilities for prompt versioning while providing a unified interface to multiple AI providers.

## ✨ Key Features

- **📚 Centralized Prompt Management**: Store, version, and track prompts in MLflow
- **🧠 Intelligent Prompt Matching**: Automatically match user queries to optimal templates
- **🔄 Dynamic Parameter Extraction**: Identify and extract parameters from natural language
- **🔍 Content Type Detection**: Recognize the type of content being requested
- **🛠️ Multiple AI Providers**: Seamless integration with OpenAI and Ollama
- **📊 Detailed Response Tracing**: Track prompt optimization and transformation stages
- **📝 Version Control**: Track prompt history with production and archive aliases
- **🧩 Extensible Framework**: Add new prompt types without code changes

## 🚀 Installation

### Basic Installation

```bash
# Install from PyPI
pip install core4ai
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/iRahulPandey/core4ai.git
cd core4ai

# Install in development mode
pip install -e ".[dev]"
```

## ⚙️ Initial Configuration

Core4AI can be configured either through the CLI setup wizard or programmatically via the Python API.

### CLI Setup (Recommended for First-Time Setup)

```bash
# Run the setup wizard
core4ai setup
```

The wizard will guide you through:
1. **MLflow Configuration**: Enter the URI of your MLflow server (default: http://localhost:8080)
2. **Existing Prompts Import**: Import existing prompts from MLflow if needed
3. **AI Provider Selection**: Choose between OpenAI or Ollama and configure the selected provider
4. **Analytics Configuration**: Enable/disable usage tracking and analytics
5. **Sample Prompts**: Register built-in sample prompt templates

### Python API Setup

```python
from core4ai import Core4AI

# Create a Core4AI instance
ai = Core4AI()

# Set MLflow URI
ai.set_mlflow_uri("http://localhost:8080")

# Configure OpenAI
api_key = os.environ.get("OPENAI_API_KEY")  # Get from environment
ai.configure_openai(api_key=api_key, model="gpt-3.5-turbo")

# Or configure Ollama
# ai.configure_ollama(uri="http://localhost:11434", model="llama3.2:latest")

# Save the configuration
ai.save_config()
```

## 📝 Prompt Management

Core4AI uses a powerful prompt management system that allows you to create, register, and use prompt templates in various formats.

### Prompt Template Format

Core4AI uses markdown files to define prompt templates. Each template should follow this structure:

```markdown
# Prompt Name: example_prompt

## Description
A brief description of what this prompt does.

## Tags
type: example
task: writing
purpose: demonstration

## Template
Write a {{ style }} response about {{ topic }} that includes:
- Important point 1
- Important point 2
- Important point 3

Please ensure the tone is {{ tone }} and suitable for {{ audience }}.
```

#### Key Guidelines

1. **Prompt Name** is required and must:
   - Be the first line of the file
   - Use the format `# Prompt Name: name_prompt`
   - End with `_prompt` suffix
   - Use underscores instead of spaces (e.g., `cover_letter_prompt`)

2. **Template Section** must:
   - Use double braces for variables: `{{ variable_name }}`
   - Have at least one variable
   - Provide clear instructions

3. **Tags Section** is recommended and should include:
   - `type`: The prompt category (e.g., essay, email, code)
   - `task`: The purpose (e.g., writing, analysis, instruction)
   - Additional metadata as needed

### Creating Prompt Templates

#### CLI Approach

```bash
# Create a new prompt template in the current directory
core4ai register --create email

# Create a prompt template in a specific directory
core4ai register --create blog --dir ./my_prompts
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# Create a new prompt template in the current directory
result = ai.create_prompt_template("email")

# Create a prompt template in a specific directory
result = ai.create_prompt_template("blog", output_dir="./my_prompts")
```

This will:
1. Create a template file with the proper structure
2. Open it in your default editor for customization
3. Offer to register it immediately after editing

### Registering Prompts

#### CLI Approach

```bash
# Register a single prompt directly
core4ai register --name "email_prompt" "Write a {{ formality }} email..."

# Register from a markdown file
core4ai register --markdown ./my_prompts/email_prompt.md

# Register all prompts from a directory
core4ai register --dir ./my_prompts

# Register built-in sample prompts
core4ai register --samples

# Register only prompts that don't exist yet
core4ai register --dir ./my_prompts --only-new
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# Register a single prompt directly
ai.register_prompt(
    name="email_prompt",
    template="Write a {{ formality }} email...",
    tags={"type": "email", "task": "writing"}
)

# Register from a markdown file
ai.register_from_markdown("./my_prompts/email_prompt.md")

# Register built-in sample prompts
ai.register_samples()

# Register from a JSON file
ai.register_from_file("./my_prompts.json")
```

### Managing Prompt Types

Core4AI automatically tracks prompt types based on the prompt names:

#### CLI Approach

```bash
# List all registered prompt types
core4ai list-types
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# List all registered prompt types
prompt_types = ai.list_prompt_types()
print(prompt_types)

# Add a new prompt type
ai.add_prompt_type("custom_type")
```

The type is extracted from the prompt name:
- For `email_prompt` → type is `email`
- For `cover_letter_prompt` → type is `cover_letter`

### Listing Available Prompts

#### CLI Approach

```bash
# List all prompts
core4ai list

# Show detailed information
core4ai list --details

# Get details for a specific prompt
core4ai list --name email_prompt@production
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# List all prompts
prompts = ai.list_prompts()
print(prompts)

# Get the configuration
config = ai.get_current_config()
print(config)
```

## 🛠️ Using Core4AI

### Basic Chat Interactions

#### CLI Approach

```bash
# Simple query - Core4AI will match to the best prompt template
core4ai chat "Write about the future of AI"

# Get a simple response without enhancement details
core4ai chat --simple "Write an essay about climate change"

# See verbose output with prompt enhancement details
core4ai chat --verbose "Write an email to my boss about a vacation request"
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# Simple query
response = ai.chat("Write about the future of AI")
print(response["response"])

# With verbose output
response = ai.chat("Write an email to my boss about a vacation request", verbose=True)
print(response["prompt_match"])  # Show matched prompt details
print(response["enhanced_query"])  # Show enhanced query
print(response["response"])  # Show final response
```

### Sample Prompts

Core4AI comes with several pre-registered prompt templates:

#### CLI Approach

```bash
# Register sample prompts
core4ai register --samples
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# Register sample prompts
result = ai.register_samples()
print(f"Registered {result.get('registered', 0)} prompts")
```

This will register the following prompt types:

| Prompt Type | Description | Sample Variables |
|-------------|-------------|------------------|
| `essay_prompt` | Academic writing | topic |
| `email_prompt` | Email composition | formality, recipient_type, topic, tone |
| `technical_prompt` | Technical explanations | topic, audience |
| `creative_prompt` | Creative writing | genre, topic |
| `code_prompt` | Code generation | language, task, requirements |
| `cover_letter_prompt` | Cover letter writing | position, company, experience_years |
| `qa_prompt` | Question answering | topic, tone, formality |
| `tutorial_prompt` | Step-by-step guides | level, task, tool_or_method |
| `marketing_prompt` | Marketing content | content_format, product_or_service, target_audience |
| `report_prompt` | Report generation | length, report_type, topic |
| `social_media_prompt` | Social media posts | number, platform, topic |
| `data_analysis_prompt` | Data analysis reports | data_type, subject, data |
| `comparison_prompt` | Compare items or concepts | item1, item2 |
| `product_description_prompt` | Product descriptions | length, product_name, product_category |
| `summary_prompt` | Content summarization | length, content_type, content |
| `research_prompt` | Research analysis | topic, tone, audience_expertise |
| `interview_prompt` | Interview preparation | position_title, company_type, experience_level |
| `syllabus_prompt` | Learning syllabi | subject, audience, duration |

Each prompt is designed for specific use cases and includes variables that can be automatically extracted from user queries. You can view the details of any prompt with:

#### CLI Approach

```bash
# View details of a specific prompt
core4ai list --name essay_prompt@production --details
```

#### Python API Approach

```python
from core4ai.prompt_manager.registry import get_prompt_details

# Get details of a specific prompt
details = get_prompt_details("essay_prompt@production")
print(details)
```

## 📊 Prompt Analytics

Core4AI includes a powerful analytics system that helps you track and analyze prompt usage to optimize your workflows.

### Analytics Features

- **Usage Tracking**: Record every use of a prompt with performance metrics
- **Provider Analysis**: See which AI providers perform best with different prompts
- **Temporal Analysis**: Track prompt usage over time
- **Performance Metrics**: Measure confidence scores, processing times, and success rates

### Using Analytics

#### CLI Approach

```bash
# View analytics for all prompts
core4ai analytics prompt

# View analytics for a specific prompt
core4ai analytics prompt --name email_prompt

# View analytics for the last 30 days
core4ai analytics prompt --time-range 30

# View overall usage statistics
core4ai analytics usage

# Export analytics data to JSON
core4ai analytics prompt --output analytics.json

# Clear analytics data
core4ai analytics clear

# Generate a dashboard in current directory
core4ai analytics dashboard
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# Get analytics for all prompts
all_analytics = ai.get_prompt_analytics()
print(f"Total prompts tracked: {len(all_analytics['metrics'])}")

# Get analytics for a specific prompt
email_analytics = ai.get_prompt_analytics("email_prompt")
if email_analytics["metrics"]:
    print(f"Email prompt used {email_analytics['metrics'][0]['total_uses']} times")

# Get overall usage statistics
usage_stats = ai.get_usage_stats(time_range=30)  # Last 30 days
print(f"Total usage in last 30 days: {usage_stats['total_count']}")

# Clear analytics for a specific prompt
ai.clear_analytics("test_prompt")

# Generate a dashboard with default settings
dashboard_path = ai.dashboard()
print(f"Dashboard saved to: {dashboard_path}")
```

### Analytics Configuration

You can configure analytics during setup or programmatically:

```python
from core4ai import Core4AI

# initialize
ai = Core4AI()

# Enable analytics
ai.configure_analytics(enabled=True)

# Disable analytics
ai.configure_analytics(enabled=False)

# Set custom database location
ai.configure_analytics(enabled=True, db_path="/path/to/analytics.db")
```

## 🔄 Provider Configuration

### OpenAI

#### CLI Approach

```bash
# Set environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"

# Or configure during setup
core4ai setup
# Choose OpenAI and follow the prompts
```

#### Python API Approach

```python
from core4ai import Core4AI
import os

ai = Core4AI()

# Using environment variable (recommended)
ai.configure_openai(model="gpt-3.5-turbo")

# Or with explicit API key (less secure)
api_key = "your-api-key-here"
ai.configure_openai(api_key=api_key, model="gpt-4")

# Save configuration (API key won't be saved to disk)
ai.save_config()
```

Available models include:
- `gpt-3.5-turbo` (default)
- `gpt-4`
- `gpt-4-turbo`
- `gpt-4o`

### Ollama

#### CLI Approach

```bash
# Install and start Ollama
ollama serve

# Configure Core4AI
core4ai setup
# Choose Ollama and follow the prompts
```

#### Python API Approach

```python
from core4ai import Core4AI

ai = Core4AI()

# Configure to use Ollama
ai.configure_ollama(uri="http://localhost:11434", model="llama2")

# Save configuration
ai.save_config()
```

## 📋 Command Reference

| Command | Description | CLI Example | Python API Example |
|---------|-------------|-------------|-------------------|
| Setup | Configure Core4AI | `core4ai setup` | `ai = Core4AI()`<br>`ai.set_mlflow_uri("http://localhost:8080")`<br>`ai.configure_openai()` |
| Register | Register prompts | `core4ai register --samples` | `ai.register_samples()` |
| List | List available prompts | `core4ai list --details` | `ai.list_prompts()` |
| List Types | List prompt types | `core4ai list-types` | `ai.list_prompt_types()` |
| Chat | Chat with enhanced prompts | `core4ai chat "Write about AI"` | `response = ai.chat("Write about AI")`<br>`print(response["response"])` |
| Analytics | View prompt analytics | `core4ai analytics prompt` | `ai.get_prompt_analytics()` |
| Version | Show version info | `core4ai version` | `from core4ai import __version__`<br>`print(__version__)` |

## 📊 How Core4AI Works

Core4AI follows this workflow to process queries:

1. **Query Analysis**: Analyze the user's query to determine intent
2. **Prompt Matching**: Match the query to the most appropriate prompt template
3. **Parameter Extraction**: Extract relevant parameters from the query
4. **Template Application**: Apply the template with extracted parameters
5. **Validation**: Validate the enhanced prompt for completeness and accuracy
6. **Adjustment**: Adjust the prompt if validation issues are found
7. **AI Response**: Send the optimized prompt to the AI provider
8. **Analytics**: Track usage metrics and performance statistics

### From Query to Enhanced Response
The user experience with Core4AI is straightforward yet powerful:

![alt text](user-flow.png)

This workflow ensures that every user query is intelligently matched to the optimal prompt template stored in MLflow, parameters are properly extracted and applied, and the result is validated before being sent to the AI provider.

## Troubleshooting Installation

### NumPy Binary Incompatibility

If you encounter an error like this during installation or when running Core4AI:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

Try reinstalling in the following order:

```bash
# Remove the problematic packages
pip uninstall -y numpy pandas mlflow core4ai

# Reinstall in the correct order with specific versions
pip install numpy==1.26.0
pip install pandas
pip install mlflow>=2.21.0
pip install core4ai
```

### MLflow Server Connection Issues

If you encounter problems connecting to MLflow:

1. Make sure your MLflow server is running:
   ```bash
   mlflow server --host 0.0.0.0 --port 8080
   ```

2. Verify connection:
   ```bash
   curl http://localhost:8080
   ```

3. Configure Core4AI to use your MLflow server:
   ```bash
   core4ai setup
   ```
   
   Or with Python:
   ```python
   from core4ai import Core4AI
   ai = Core4AI()
   ai.set_mlflow_uri("http://localhost:8080")
   ai.save_config()
   ```

### Analytics Issues

If you experience issues with analytics:

1. Check if analytics is enabled:
   ```bash
   core4ai version
   ```
   The output will show analytics status.

2. Verify database location:
   ```bash
   ls -la ~/.core4ai/analytics.db
   ```
   
3. Reset analytics if needed:
   ```bash
   core4ai analytics clear
   ```
   
## 📜 License

This project is licensed under the Apache License 2.0