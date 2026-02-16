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
  docstrings. Supports multiple Qwen Coder models with model-specific configurations.

- **`models.py`** - Model configuration registry with sampling parameters for
  Qwen 2.5 Coder and Qwen3 Coder variants.

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
2. **Pull a model**: Download one of the supported code models:
   ```bash
   # Qwen 2.5 Coder (dense models)
   ollama pull qwen2.5-coder:32b  # Default, ~18GB Q4
   ollama pull qwen2.5-coder:14b  # Mid-size, ~8GB Q4
   ollama pull qwen2.5-coder:7b   # Fast, ~4GB Q4

   # Qwen3 Coder (MoE model)
   ollama pull qwen3-coder:30b-a3b  # Best quality, ~18GB Q4, 256K context
   ```

### Starting the Server

Start the FastAPI server using uvicorn:

**Linux/macOS:**
```bash
# Using uvicorn directly
uvicorn src.training.serve:app --host 0.0.0.0 --port 8000

# Or run the module directly
python -m src.training.serve
```

**Windows (PowerShell):**
```powershell
uvicorn src.training.serve:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000` by default.

### Configuration

The server can be configured using environment variables:

- `OLLAMA_URL` - Ollama API endpoint (default: `http://localhost:11434/api/chat`)
- `OLLAMA_MODEL` - Model key or Ollama model name (default: `qwen2.5-coder-32b`)
- `REQUEST_TIMEOUT` - Request timeout in seconds (default: `120.0`)

**Linux/macOS:**
```bash
OLLAMA_MODEL=qwen3-coder-30b uvicorn src.training.serve:app --port 8000
```

**Windows (PowerShell):**
```powershell
$env:OLLAMA_MODEL="qwen3-coder-30b"; uvicorn src.training.serve:app --port 8000
```

**Windows (CMD):**
```cmd
set OLLAMA_MODEL=qwen3-coder-30b && uvicorn src.training.serve:app --port 8000
```

### Available Models

| Model Key | Ollama Model | Architecture | Memory (Q4) | Context | Description |
|-----------|--------------|--------------|-------------|---------|-------------|
| `qwen2.5-coder-32b` | `qwen2.5-coder:32b` | Dense | ~18GB | 32K | Default, balanced quality/speed |
| `qwen2.5-coder-14b` | `qwen2.5-coder:14b` | Dense | ~8GB | 32K | Mid-size, good performance |
| `qwen2.5-coder-7b` | `qwen2.5-coder:7b` | Dense | ~4GB | 32K | Fast inference |
| `qwen3-coder-30b` | `qwen3-coder:30b-a3b` | MoE | ~18GB | 256K | Best quality, 3.3B active params |

Each model has optimized sampling parameters:
- **Qwen 2.5 Coder**: temperature=0.7, top_p=0.9, top_k=40
- **Qwen3 Coder**: temperature=1.0, top_p=0.95, top_k=40 (per official recommendations)

### Model Selection

You can select a model in two ways:

1. **Environment variable** (applies to all requests):
   ```bash
   OLLAMA_MODEL=qwen3-coder-30b uvicorn src.training.serve:app
   ```

2. **Per-request** (via API):
   ```bash
   curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"code": "def add(x, y): return x + y", "model": "qwen3-coder-30b"}'
   ```

### List Available Models

**Via CLI:**
```bash
python scripts/run_ollama.py --list-models
```

**Via API:**
```bash
curl http://localhost:8000/models
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
  "service": "ollama",
  "active_model": "Qwen 2.5 Coder 32B",
  "ollama_model": "qwen2.5-coder:32b"
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
- `max_new_tokens` (optional): Maximum number of tokens to generate (uses model default if not specified)
- `model` (optional): Model key or Ollama model name to use for this request

**Response (200 OK):**
```json
{
  "docstring": "\"\"\"Compute the sum of two numbers.\n\nParameters\n----------\nx : int\n    First number.\ny : int\n    Second number.\n\nReturns\n-------\nint\n    Sum of x and y.\"\"\"",
  "model": "qwen2.5-coder:32b"
}
```

**Response (500 Internal Server Error):**
```json
{
  "detail": "Failed to generate docstring: <error message>"
}
```

#### List Models

Get available model configurations:

```bash
curl http://localhost:8000/models
```

**Response (200 OK):**
```json
{
  "default": "qwen2.5-coder-32b",
  "active": "qwen2.5-coder-32b",
  "models": [
    {
      "key": "qwen2.5-coder-32b",
      "name": "Qwen 2.5 Coder 32B",
      "ollama_model": "qwen2.5-coder:32b",
      "context_window": 32768,
      "architecture": "dense",
      "memory_q4": "~18GB",
      "description": "Dense 32B model, good balance of quality and speed"
    }
  ]
}
```

### CLI Tool

The CLI tool allows testing docstring generation directly:

```bash
# Use default model
python scripts/run_ollama.py --user "def add(x, y): return x + y"

# Use specific model by key
python scripts/run_ollama.py --model-key qwen3-coder-30b --user "def foo(): pass"

# Use raw Ollama model name
python scripts/run_ollama.py --model qwen2.5-coder:7b --user "def bar(): pass"

# List available models
python scripts/run_ollama.py --list-models
```

### Testing

Run the test suite to verify the API endpoints:

```bash
pytest tests/test_serve.py tests/test_models.py -v
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
