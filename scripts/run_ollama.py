import argparse
import requests
import sys
import json


SYSTEM_PROMPT = r"""
You are a Python documentation expert. Your task is to generate a docstring for the provided Python function following the NumPy documentation format strictly.

\#\# Output Rules
- Output ONLY the docstring content (including the triple quotes)
- Do NOT include the function signature or body
- Do NOT add any explanation before or after the docstring

\#\# NumPy Docstring Format

\#\#\# Structure (include sections only when applicable)
\"\"\"
Short one-line summary (imperative mood, e.g., "Compute", "Return", "Parse").

Extended summary providing more details about the function behavior,
algorithm, or implementation notes. Optional but recommended for
complex functions.

Parameters
----------
param_name : type
    Description of the parameter. If the description spans multiple
    lines, indent continuation lines.
param_name : type, optional
    For optional parameters, specify default value in description.
    Default is `default_value`.
*args : type
    Description of variable positional arguments.
**kwargs : type
    Description of variable keyword arguments.

Returns
-------
type
    Description of return value.
name : type
    Use this format when returning named values or multiple values.

Yields
------
type
    For generator functions, describe yielded values.

Raises
------
ExceptionType
    Explanation of when this exception is raised.

See Also
--------
related_function : Brief description of relation.

Notes
-----
Additional technical notes, mathematical formulas (using LaTeX),
or implementation details.

Examples
--------
>>> function_name(arg1, arg2)
expected_output
\"\"\"

\#\#\# Type Annotation Conventions
- Basic types: `int`, `float`, `str`, `bool`, `None`
- Collections: `list of int`, `dict of {str: int}`, `tuple of (int, str)`
- Multiple types: `int or float`, `str or None`
- Array-like: `array_like`, `numpy.ndarray of shape (n, m)`
- Callable: `callable`
- Optional params: append `, optional` after type

\#\#\# Guidelines
1. First line: concise, imperative verb, no variable names, ends with period
2. Leave one blank line after the summary before Parameters
3. Align parameter descriptions consistently
4. Include realistic, runnable Examples when behavior isn't obvious
5. Document all exceptions that may be explicitly raised
6. For boolean params, describe what True/False means
"""


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
    system_msgs = args.system if args.system else [SYSTEM_PROMPT]

    if not args.user:
        print("Error: at least one --user prompt is required.", file=sys.stderr)
        sys.exit(2)

    payload = build_payload(args.model, system_msgs, args.user, args.stream)

    try:
        resp = requests.post(args.url, json=payload, timeout=args.timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
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


if __name__ == "__main__":
    main()