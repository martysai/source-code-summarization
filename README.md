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

- **`serve.py`** - FastAPI inference server that loads the fine-tuned model and
  serves docstring generation via HTTP.

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

# Download the seed dataset
cd data/raw/python-method && bash get_data.sh && cd -

# Convert to HuggingFace format
python -m src.data.convert_seed \
    --input-dir data/raw/python-method \
    --output-dir data/processed/python-method
```

## Dataset

The seed dataset comes from the [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum)
project (ACL 2020): 92,545 Python function-docstring pairs split into train/dev/test.

## Acknowledgments

- Original C2NL dataset: [A Transformer-based Approach for Source Code Summarization](https://arxiv.org/abs/2005.00653)
- Python150k dataset: [ETH Zurich SRI Lab](https://www.sri.inf.ethz.ch/py150)
- Tree Transformer: [nxphi47/tree_transformer](https://github.com/nxphi47/tree_transformer)
