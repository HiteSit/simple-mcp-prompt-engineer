# Simple MCP Prompt Engineer

This project implements a powerful prompt optimization server using the Model Context Protocol (MCP). It provides a systematic approach to improving AI prompts through multiple stages of optimization, including analysis, rules application, structuring, verification, and refinement.

## Features

- 🔍 **Smart Prompt Analysis**: Identifies prompt types and opportunities for improvement
- 🔧 **Rule-Based Optimization**: Applies best practices for prompt engineering
- 📋 **Structured Output**: Improves organization and clarity of prompts
- 🔄 **Iterative Refinement**: Allows further improvements based on user feedback
- 📊 **Optimization History**: Tracks the evolution of prompts over multiple versions

## Prerequisites

- Python 3.12 or higher
- Poetry for dependency management

## Project Structure

```
simple-mcp-prompt-engineer/
├── simple-mcp_prompt_engineer/
│   ├── server.py
│   └── __init__.py
├── README.md
└── pyproject.toml
```

## Quick Start

1. **Set Up Project**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/simple-mcp-prompt-engineer.git
   cd simple-mcp-prompt-engineer
   
   # Install dependencies with Poetry
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

2. **Run the Server**
   ```bash
   # From the project root directory
   python -m simple-mcp_prompt_engineer.server
   ```

## Claude Desktop Integration

Add to your Claude Desktop configuration (`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "prompt-engineer": {
      "command": "python",
      "args": [
        "-m",
        "simple-mcp_prompt_engineer.server"
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

## Troubleshooting

Common issues:

- **Server Connection Issues**
  - Verify paths in claude_desktop_config.json
  - Check Claude Desktop logs: `%APPDATA%\Claude\logs`
  - Ensure Python 3.12+ is installed

## License

MIT License

## Acknowledgments

This project is totally inspired by the [Model Context Protocol](https://github.com/modelcontextprotocol/servers) repository and framework. Their pioneering work on creating standardized protocols for AI model interactions has made projects like this possible.

## Author

Riccardo Fusco
