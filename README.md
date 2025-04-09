# Simple MCP Prompt Engineer

This project implements a powerful prompt optimization server using the Model Context Protocol (MCP). It provides a systematic approach to improving AI prompts through multiple stages of optimization, including analysis, rules application, structuring, verification, and refinement.

## Features

- 🔍 **Smart Prompt Analysis**: Identifies prompt types and opportunities for improvement
- 🔧 **Rule-Based Optimization**: Applies best practices for prompt engineering
- 📋 **Structured Output**: Improves organization and clarity of prompts
- 🔄 **Iterative Refinement**: Allows further improvements based on user feedback
- 📊 **Optimization History**: Tracks the evolution of prompts over multiple versions

## Setup

  ```bash
  # Create and activate virtual environment
  uv venv
  .venv\Scripts\activate     # Windows
  source .venv/bin/activate  # Unix

  # Install dependencies
  uv pip install rich mcp-server
  ```

## Project Structure

```
simple-mcp-prompt-engineer/
├── simple-mcp_prompt_engineer/
│   ├── server.py
│   └── __init__.py
├── README.md
```

## Claude Desktop Integration

Add to your Claude Desktop configuration.

```json
{
  "mcpServers": {
    "prompt-engineer": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Path\\to\\simple-mcp_prompt_engineer",  \\ On linux: /path/to/simple-mcp_prompt_engineer
        "run",
        "server.py"
      ]
    }
  }
}
```

## API

The server exposes four main tools:

### 1. `optimize_prompt`

Automatically optimizes a prompt based on best practices.

Parameters:
- `prompt` (str): The original prompt to optimize

Returns:
- Optimized prompt

### 2. `refine_prompt`

Refines the current prompt based on user feedback.

Parameters:
- `feedback` (str): User feedback for further refinement

Returns:
- Refined prompt

### 3. `get_optimization_history`

Get the full optimization history.

Returns:
- JSON string containing the optimization history

### 4. `clear_optimization_history`

Clear the optimization history.

Returns:
- Confirmation message

## Optimization Process

The prompt optimization goes through the following stages:

1. **Initial Analysis**: Detecting prompt type and structure
2. **Rules Application**: Applying best practices for prompt engineering
3. **Structuring**: Organizing content into clear sections
4. **Verification**: Ensuring all important context is preserved
5. **Refinement**: Applying user feedback for further improvements
6. **Final**: Polished, optimized prompt ready for use

## License

MIT License

## Acknowledgments

This project is totally inspired by the [Model Context Protocol](https://github.com/modelcontextprotocol/servers) repository and framework. Their pioneering work on creating standardized protocols for AI model interactions has made projects like this possible.

## Author

Riccardo Fusco
