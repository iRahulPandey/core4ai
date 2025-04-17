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

After installation, run the interactive setup wizard:

```bash
# Run the setup wizard
core4ai setup
```


The wizard will guide you through:

1. **MLflow Configuration**: 
   - Enter the URI of your MLflow server (default: http://localhost:8080)
   - Core4AI will use MLflow to store and version your prompts

2. **Existing Prompts Import**:
   - If you already have prompts in MLflow, you can import them into Core4AI
   - You can provide prompt names directly or via a file

3. **AI Provider Selection**:
   - Choose between OpenAI or Ollama
   - For OpenAI: Set your API key as an environment variable:
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```
   - For Ollama: Specify the server URI and model to use

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

You can create a new prompt template using the CLI:

```bash
# Create a new prompt template in the current directory
core4ai register --create email

# Create a prompt template in a specific directory
core4ai register --create blog --dir ./my_prompts
```

This will:
1. Create a template file with the proper structure
2. Open it in your default editor for customization
3. Offer to register it immediately after editing

### Registering Prompts

Core4AI supports multiple ways to register prompts:

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

### Managing Prompt Types

Core4AI automatically tracks prompt types based on the prompt names:

```bash
# List all registered prompt types
core4ai list-types
```

The type is extracted from the prompt name:
- For `email_prompt` → type is `email`
- For `cover_letter_prompt` → type is `cover_letter`

### Listing Available Prompts

```bash
# List all prompts
core4ai list

# Show detailed information
core4ai list --details

# Get details for a specific prompt
core4ai list --name email_prompt
```

## 🛠️ Using Core4AI

### Basic Chat Interactions

```bash
# Simple query - Core4AI will match to the best prompt template
core4ai chat "Write about the future of AI"

# Get a simple response without enhancement details
core4ai chat --simple "Write an essay about climate change"

# See verbose output with prompt enhancement details
core4ai chat --verbose "Write an email to my boss about a vacation request"
```

### Sample Prompts

Core4AI comes with several pre-registered prompt templates:

```bash
# Register sample prompts
core4ai register --samples
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
| `test_prompt` | Test examples | formality |

Each prompt is designed for specific use cases and includes variables that can be automatically extracted from user queries. You can view the details of any prompt with:

```bash
# View details of a specific prompt
core4ai list --name email_prompt@production --details
```

## 🔄 Provider Configuration

### OpenAI

To use OpenAI, set your API key:

```bash
# Set environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"

# Or configure during setup
core4ai setup
```

Available models include:
- `gpt-3.5-turbo` (default)
- `gpt-4`
- `gpt-4-turbo`

### Ollama

To use Ollama:

1. [Install Ollama](https://ollama.ai/download) on your system
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Configure Core4AI:
   ```bash
   core4ai setup
   ```

## 📋 Command Reference

| Command | Description | Examples |
|---------|-------------|----------|
| `core4ai setup` | Run the setup wizard | `core4ai setup` |
| `core4ai register` | Register prompts | `core4ai register --samples`, `core4ai register --create email` |
| `core4ai list` | List available prompts | `core4ai list --details` |
| `core4ai list-types` | List prompt types | `core4ai list-types` |
| `core4ai chat` | Chat with enhanced prompts | `core4ai chat "Write about AI"` |
| `core4ai version` | Show version info | `core4ai version` |

## 📊 How Core4AI Works

Core4AI follows this workflow to process queries:

1. **Query Analysis**: Analyze the user's query to determine intent
2. **Prompt Matching**: Match the query to the most appropriate prompt template
3. **Parameter Extraction**: Extract relevant parameters from the query
4. **Template Application**: Apply the template with extracted parameters
5. **Validation**: Validate the enhanced prompt for completeness and accuracy
6. **Adjustment**: Adjust the prompt if validation issues are found
7. **AI Response**: Send the optimized prompt to the AI provider

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

## 📜 License

This project is licensed under the Apache License 2.0