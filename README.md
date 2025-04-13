# Core4AI: Contextual Optimization and Refinement Engine for AI

Core4AI is an intelligent system that transforms basic user queries into optimized prompts for AI systems using MLflow Prompt Registry. It dynamically matches user requests to the most appropriate prompt template and applies it with extracted parameters.

## 🔍 Overview

Core4AI combines MLflow Prompt Registry with dynamic prompt matching to create a powerful, flexible system for prompt engineering:

- **Centralized Prompt Management**: Store, version, and manage prompts in MLflow
- **Dynamic Matching**: Intelligently match user queries to the best prompt template
- **Multiple AI Providers**: Support for OpenAI and Ollama
- **Version Control**: Track prompt history with production and archive aliases
- **Extensible**: Easily add new prompt types without code changes

## 🚀 Installation

```bash
# Install the package
pip install core4ai
```

## ⚙️ Initial Setup

After installation, run the interactive setup wizard:

```bash
# Run the setup wizard
core4ai setup
```

The wizard will guide you through:

1. **MLflow Configuration**: Enter the URI of your local MLflow server
2. **AI Provider Selection**: Choose between OpenAI or Ollama
   - For OpenAI: Set your API key as an environment variable
   - For Ollama: Specify the server URI and model to use

## 🛠️ Usage

### Starting the Server

```bash
# Start the Core4AI server
core4ai serve
```

### Managing Prompts

```bash
# Register a new prompt
core4ai register --name "essay_prompt" --message "Basic essay template" "Write a well-structured essay on {{ topic }} that includes an introduction, body paragraphs, and conclusion."

# Register prompts from a JSON file
core4ai register --file prompts.json

# List available prompts
core4ai list
```

### Chatting with AI

```bash
# Basic chat query
core4ai chat "Write about the future of AI"

# Verbose mode shows prompt enhancement details
core4ai chat --verbose "Write an email to my boss about a vacation request"
```

## 📋 Available Commands

- `core4ai setup`: Run the interactive setup wizard
- `core4ai serve`: Start the Core4AI server
- `core4ai register`: Register a new prompt or prompts from a file
- `core4ai list`: List all available prompts
- `core4ai chat`: Chat with AI using enhanced prompts

## 🧩 Supported Prompt Types

Core4AI supports various prompt types, including:

- `essay_prompt`: Academic writing
- `email_prompt`: Email composition
- `technical_prompt`: Technical explanations
- `creative_prompt`: Creative writing
- `code_prompt`: Code generation
- `summary_prompt`: Content summarization
- ... and more!

## 🔄 Provider Configuration

### OpenAI

To use OpenAI, set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Ollama

To use Ollama:

1. Ensure your Ollama server is running
2. During setup, provide:
   - Server URI (default: http://localhost:11434)
   - Model name (e.g., llama2)

## 📦 Project Structure

```
core4ai/
├── cli/              # Command-line interface
├── server/           # Server functionality
├── client/           # Client implementation
├── prompt_manager/   # MLflow integration
├── providers/        # AI provider implementations
└── config/           # Configuration management
```

## 🔧 Troubleshooting

- **MLflow Connection Issues**: Ensure your MLflow server is running and accessible at the configured URI
- **OpenAI API Key**: Make sure your OpenAI API key is set as an environment variable
- **Ollama Server**: Verify that your Ollama server is running and the model is available

For more help, use the verbose flag:

```bash
core4ai --verbose [command]
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.