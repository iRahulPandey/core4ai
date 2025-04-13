# Core4AI: Contextual Optimization and Refinement Engine for AI

Core4AI is an intelligent system that transforms basic user queries into optimized prompts for AI systems using MLflow Prompt Registry. It dynamically matches user requests to the most appropriate prompt template and applies it with extracted parameters.

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

2. **AI Provider Selection**:
   - Choose between OpenAI or Ollama
   - For OpenAI: Set your API key as an environment variable:
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```
   - For Ollama: Specify the server URI and model to use

## 🛠️ Usage Examples

### Managing Prompts

```bash
# Register a new prompt with a template
core4ai register --name "essay_prompt" \
    --message "Basic essay template" \
    "Write a well-structured essay on {{ topic }} that includes an introduction, body paragraphs, and conclusion."

# Register multiple prompts from a JSON file
core4ai register --file prompts.json

# List all available prompts
core4ai list

# Get details about a specific prompt
core4ai list --name "essay_prompt@production" --details

# Later, update the same prompt (creates a new version)
core4ai register --name "email_prompt" "Write an updated {{ formality }} email..."
```

### Basic Chat Interactions

```bash
# Simple query - Core4AI will match to the best prompt template
core4ai chat "Write about the future of AI"

# Get a simple response without enhancement details
core4ai chat --simple "Write an essay about climate change"

# See verbose output with prompt enhancement details
core4ai chat --verbose "Write an email to my boss about a vacation request"
```

## 💡 Sample Prompts

Core4AI comes with several pre-registered prompt templates:

```bash
# Register sample prompts
core4ai register --samples
```

This will register the following prompt types:

| Prompt Type | Description | Sample Variables |
|-------------|-------------|------------------|
| `essay_prompt` | Academic writing | topic |
| `email_prompt` | Email composition | formality, recipient_type, topic |
| `technical_prompt` | Technical explanations | topic, audience |
| `creative_prompt` | Creative writing | genre, topic |
| `code_prompt` | Code generation | language, task, requirements |
| `summary_prompt` | Content summarization | content, length |

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
| `core4ai register` | Register a new prompt | `core4ai register --name "email_prompt" "Write a {{ formality }} email..."` |
| `core4ai list` | List available prompts | `core4ai list --details` |
| `core4ai chat` | Chat with enhanced prompts | `core4ai chat "Write about AI"` |
| `core4ai version` | Show version info | `core4ai version` |

## 🧪 Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test category
pytest tests/unit
pytest tests/functional
pytest tests/integration
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 How Core4AI Works

Core4AI follows this workflow to process queries:

1. **Query Analysis**: Analyze the user's query to determine intent
2. **Prompt Matching**: Match the query to the most appropriate prompt template
3. **Parameter Extraction**: Extract relevant parameters from the query
4. **Template Application**: Apply the template with extracted parameters
5. **Validation**: Validate the enhanced prompt for completeness and accuracy
6. **Adjustment**: Adjust the prompt if validation issues are found
7. **AI Response**: Send the optimized prompt to the AI provider

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
   curl http://localhost:8080/api/2.0/mlflow/experiments/list
   ```

3. Configure Core4AI to use your MLflow server:
   ```bash
   core4ai setup
   ```

## 📜 License

This project is licensed under the Apache License 2.0