import argparse
import requests
import sys
import json
import time
from pathlib import Path


def load_system_prompt() -> str:
    """Load the default system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "src" / "training" / "prompts" / "system_prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {prompt_path}. "
            "Please ensure the prompt file exists."
        )
    return prompt_path.read_text(encoding="utf-8")


DEFAULT_URL = "http://localhost:11434/api/chat"  # Changed from /api/generate


def build_payload(model, system_msgs, user_msgs, stream):
    messages = []
    for s in system_msgs:
        messages.append({"role": "system", "content": s})
    for u in user_msgs:
        messages.append({"role": "user", "content": u})

    return {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": 0  # Unload model after request
    }


def main():
    parser = argparse.ArgumentParser(description="Send system and user prompts to model endpoint")
    parser.add_argument("--url", default=DEFAULT_URL, help="API endpoint URL")
    parser.add_argument("--model", required=True, help="Model name, e.g. qwen2.5-coder:32b")
    parser.add_argument("--system", action="append", default=None, help="System prompt (repeatable). Overrides default.")
    parser.add_argument("--user", action="append", default=[], help="User prompt (repeatable)")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds")

    args = parser.parse_args()

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
        sys.exit(2)

    payload = build_payload(args.model, system_msgs, args.user, args.stream)

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
    print(f"Execution time: {elapsed_time:.2f}s", file=sys.stdout)


if __name__ == "__main__":
    main()