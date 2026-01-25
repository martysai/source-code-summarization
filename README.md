# Source Code Summarization

LLM-based Python code summarization with AST-aware evaluation.

## Overview

This project fine-tunes small code LLMs (1-3B parameters) via LoRA to generate
docstrings for Python functions, and evaluates them using an AST-aware benchmark
that tests structural understanding beyond surface-level text metrics.

### Architecture

```
Seed Dataset (C2NL, 92k examples)
        |
        v
[convert_seed.py] --> HuggingFace Dataset
        |
        v
[expand_with_distilabel.py] --> Expanded Dataset (teacher LLM generates more examples)
        |
        v
[train_lora.py] --> LoRA-adapted Code LLM
        |
        v
[serve.py] --> FastAPI Inference Server (localhost:8000)
        |
        v
    VS Code Extension (calls /generate endpoint)
```

Evaluation runs independently via the AST-aware benchmark:

```
Test Dataset + Model Predictions --> [benchmark.py] --> Metrics Report
                                        |
                          Standard (BLEU, ROUGE) + AST-aware metrics
```

## Components

### Data Preparation (`src/data/`)

- **`convert_seed.py`** - Converts the C2NL parallel-file dataset (code.original +
  javadoc.original) into HuggingFace instruction-tuning format. Applies heuristic
  detokenization to make code readable for LLMs.

- **`expand_with_distilabel.py`** - Uses distilabel to expand the seed dataset by
  sending code to a teacher LLM for higher-quality docstring generation.

### Training (`src/training/`)

- **`train_lora.py`** - LoRA fine-tuning using HuggingFace Trainer + PEFT. Supports
  QLoRA (4-bit quantization) for training on 1-2 A100 GPUs.

- **`serve.py`** - FastAPI inference server that uses ollama API to generate
  docstrings. The server uses a hard-coded system prompt for NumPy-style docstring
  generation.

### Evaluation (`src/evaluation/`)

- **`benchmark.py`** - Benchmark runner that evaluates docstring quality using both
  standard and AST-aware metrics.

- **`metrics/standard.py`** - BLEU and ROUGE-L wrappers via HuggingFace evaluate.

- **`metrics/ast_aware.py`** - Novel metrics that parse the source code's AST and
  check whether generated docstrings correctly reference identifiers, control-flow
  patterns, and function parameters.

### AST Utilities (`src/ast_utils/`)

Migrated from the original Python150k preprocessing pipeline:

- **`parse_python3.py`** - Converts Python source code to a JSON AST representation.
- **`ast_conversion.py`** - Transforms AST with value-node splitting and DFS traversal.
- **`processor_ast.py`** - Text preprocessing for code, comments, and docstrings.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Convert to HuggingFace format (requires dataset access, see below)
python -m src.data.convert_seed \
    --input-dir data/raw/python-method \
    --output-dir data/processed/python-method
```

## Serving

The FastAPI inference server provides HTTP endpoints for docstring generation using
ollama as the backend. The server uses a system prompt stored in
`src/training/prompts/system_prompt.md` to generate NumPy-style docstrings.

### Prerequisites

1. **Install ollama**: Make sure [ollama](https://ollama.ai/) is installed and running locally
2. **Pull a model**: Download a code model (e.g., `qwen2.5-coder:32b`):
   ```bash
   ollama pull qwen2.5-coder:32b
   ```

### Starting the Server

Start the FastAPI server using uvicorn:

```bash
# Using uvicorn directly
uvicorn src.training.serve:app --host 0.0.0.0 --port 8000

# Or run the module directly
python -m src.training.serve
```

The server will start on `http://localhost:8000` by default.

### Configuration

The server can be configured using environment variables:

- `OLLAMA_URL` - Ollama API endpoint (default: `http://localhost:11434/api/chat`)
- `OLLAMA_MODEL` - Model name to use (default: `qwen2.5-coder:32b`)
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: `120.0`)

Example:
```bash
OLLAMA_MODEL=qwen2.5-coder:7b uvicorn src.training.serve:app --port 8000
```

### API Endpoints

#### Health Check

Check if the service is healthy and ollama is accessible:

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "ollama"
}
```

**Response (503 Service Unavailable):**
```json
{
  "detail": "Service unhealthy: ollama is not running or not accessible"
}
```

#### Generate Docstring

Generate a docstring for a Python function:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(x, y):\n    return x + y",
    "max_new_tokens": 256
  }'
```

**Request Body:**
- `code` (required): Python function code as a string
- `max_new_tokens` (optional): Maximum number of tokens to generate (default: 256)

**Response (200 OK):**
```json
{
  "docstring": "\"\"\"Compute the sum of two numbers.\n\nParameters\n----------\nx : int\n    First number.\ny : int\n    Second number.\n\nReturns\n-------\nint\n    Sum of x and y.\n\"\"\""
}
```

**Response (500 Internal Server Error):**
```json
{
  "detail": "Failed to generate docstring: <error message>"
}
```

### Testing

Run the test suite to verify the API endpoints:

```bash
pytest tests/test_serve.py -v
```

## Dataset

The seed dataset comes from the [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum)
project (ACL 2020): 92,545 Python function-docstring pairs split into train/dev/test.

### Dataset Access

The python-method dataset was previously available via a Google Drive download script
(`data/raw/python-method/get_data.sh`). This script has been removed as the Google Drive
link (file ID: `1XPE1txk9VI0aOT_TdqbAeI58Q8puKVl2`) is no longer accessible.

To obtain the dataset, you can:
1. Contact the [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum) authors
2. Download from the original source if available at the project repository
3. Use the alternative python150k dataset from [ETH Zurich SRI Lab](https://www.sri.inf.ethz.ch/py150)

## Acknowledgments

- Original C2NL dataset: [A Transformer-based Approach for Source Code Summarization](https://arxiv.org/abs/2005.00653)
- Python150k dataset: [ETH Zurich SRI Lab](https://www.sri.inf.ethz.ch/py150)
- Tree Transformer: [nxphi47/tree_transformer](https://github.com/nxphi47/tree_transformer)
