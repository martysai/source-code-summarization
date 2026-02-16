"""CLI script for testing docstring generation with Ollama.

Supports model selection via registry keys or raw Ollama model names,
with model-specific sampling parameters.

Usage:
    python scripts/run_ollama.py --model-key qwen2.5-coder-32b --user "def add(x, y): return x + y"
    python scripts/run_ollama.py --model qwen2.5-coder:7b --user "def foo(): pass"
    python scripts/run_ollama.py --list-models
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.models import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    ModelConfig,
    SamplingConfig,
    get_model_config,
    list_models,
)


DEFAULT_URL = "http://localhost:11434/api/chat"


def load_system_prompt() -> str:
    """Load the default system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "src" / "training" / "prompts" / "system_prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {prompt_path}. "
            "Please ensure the prompt file exists."
        )
    return prompt_path.read_text(encoding="utf-8")


def build_payload(
    model_config: ModelConfig,
    system_msgs: list[str],
    user_msgs: list[str],
    stream: bool
) -> dict:
    """Build the request payload with model-specific sampling parameters."""
    messages = []
    for s in system_msgs:
        messages.append({"role": "system", "content": s})
    for u in user_msgs:
        messages.append({"role": "user", "content": u})

    return {
        "model": model_config.ollama_model,
        "messages": messages,
        "stream": stream,
        "keep_alive": model_config.keep_alive,
        "options": {
            "temperature": model_config.sampling.temperature,
            "top_p": model_config.sampling.top_p,
            "top_k": model_config.sampling.top_k,
        }
    }


def print_models_list():
    """Print all available models in a formatted table."""
    print("\nAvailable Models:")
    print("=" * 80)
    print(f"{'Key':<22} {'Ollama Model':<25} {'Arch':<6} {'Memory':<8} {'Context'}")
    print("-" * 80)

    for model_info in list_models():
        key = model_info["key"]
        ollama = model_info["ollama_model"]
        arch = model_info["architecture"]
        memory = model_info.get("memory_q4", "N/A")
        context = f"{model_info['context_window']:,}"

        # Mark default model
        marker = " *" if key == DEFAULT_MODEL_KEY else ""
        print(f"{key:<22} {ollama:<25} {arch:<6} {memory:<8} {context}{marker}")

    print("-" * 80)
    print(f"* = default model ({DEFAULT_MODEL_KEY})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Send prompts to Ollama for docstring generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a registered model by key
  python scripts/run_ollama.py --model-key qwen2.5-coder-32b --user "def add(x, y): return x + y"

  # Use a raw Ollama model name
  python scripts/run_ollama.py --model qwen2.5-coder:7b --user "def foo(): pass"

  # Use Qwen3 MoE model
  python scripts/run_ollama.py --model-key qwen3-coder-30b --user "def hello(): print('hi')"

  # List available models
  python scripts/run_ollama.py --list-models
"""
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="API endpoint URL (default: %(default)s)"
    )
    parser.add_argument(
        "--model-key",
        help=f"Model key from registry (e.g., {DEFAULT_MODEL_KEY}, qwen3-coder-30b)"
    )
    parser.add_argument(
        "--model",
        help="Raw Ollama model name (e.g., qwen2.5-coder:32b). Overrides --model-key"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model configurations and exit"
    )
    parser.add_argument(
        "--system",
        action="append",
        default=None,
        help="System prompt (repeatable). Overrides default."
    )
    parser.add_argument(
        "--user",
        action="append",
        default=[],
        help="User prompt (repeatable)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: %(default)s)"
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        print_models_list()
        sys.exit(0)

    # Determine model configuration
    if args.model:
        # Raw Ollama model name provided
        model_config = get_model_config(args.model)
        print(f"Using model: {model_config.ollama_model}", file=sys.stderr)
    elif args.model_key:
        # Registry key provided
        model_config = get_model_config(args.model_key)
        print(f"Using model: {model_config.name} ({model_config.ollama_model})", file=sys.stderr)
    else:
        # Use default
        model_config = get_model_config(DEFAULT_MODEL_KEY)
        print(f"Using default model: {model_config.name} ({model_config.ollama_model})", file=sys.stderr)

    # Print sampling parameters
    print(
        f"Sampling: temp={model_config.sampling.temperature}, "
        f"top_p={model_config.sampling.top_p}, "
        f"top_k={model_config.sampling.top_k}",
        file=sys.stderr
    )

    # Use default system prompt if none provided
    if args.system:
        system_msgs = args.system
    else:
        try:
            system_msgs = [load_system_prompt()]
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.user:
        print("Error: at least one --user prompt is required.", file=sys.stderr)
        print("Use --help for usage information.", file=sys.stderr)
        sys.exit(2)

    payload = build_payload(model_config, system_msgs, args.user, args.stream)

    # Track execution time
    start_time = time.time()
    try:
        resp = requests.post(args.url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        elapsed_time = time.time() - start_time
        print(f"Request failed: {e}", file=sys.stderr)
        print(f"Execution time: {elapsed_time:.2f}s", file=sys.stderr)
        sys.exit(1)

    try:
        data = resp.json()
    except ValueError:
        print("Response is not valid JSON", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        sys.exit(1)

    # Handle /api/chat response format
    if "message" in data:
        print(data["message"].get("content", ""))
    elif "response" in data:
        print(data["response"])
    elif "choices" in data and isinstance(data["choices"], list):
        for c in data["choices"]:
            print(c.get("message", {}).get("content", c.get("text", "")))
    else:
        print(json.dumps(data, indent=2))

    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
