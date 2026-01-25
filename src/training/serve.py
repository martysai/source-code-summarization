"""FastAPI inference server for the fine-tuned docstring generation model.

Loads a LoRA-adapted model and serves predictions via HTTP. Designed to be
called by the VS Code extension for local, offline docstring generation.

Usage:
    uvicorn src.training.serve:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /generate  - Generate a docstring for a given code snippet
    GET  /health    - Health check
"""

import os
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120.0"))


def load_system_prompt() -> str:
    """Load the default system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / "system_prompt.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {prompt_path}. "
            "Please ensure the prompt file exists."
        )
    return prompt_path.read_text(encoding="utf-8")


# Load system prompt at module level
SYSTEM_PROMPT = load_system_prompt()


def generate_docstring(code: str, max_new_tokens: int = 256) -> str:
    """Generate a docstring for the given code snippet using ollama API."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": code}
        ],
        "stream": False,
        "keep_alive": 0,
        "options": {
            "num_predict": max_new_tokens
        }
    }
    
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle /api/chat response format
        if "message" in data:
            return data["message"].get("content", "")
        elif "response" in data:
            return data["response"]
        elif "choices" in data and isinstance(data["choices"], list):
            content = ""
            for c in data["choices"]:
                content += c.get("message", {}).get("content", c.get("text", ""))
            return content
        else:
            raise ValueError(f"Unexpected response format: {data}")
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to generate docstring: {e}") from e


def check_ollama_health() -> bool:
    """Check if ollama is running locally by making a test request."""
    try:
        # Try to list models as a health check
        health_url = OLLAMA_URL.replace("/api/chat", "/api/tags")
        resp = requests.get(health_url, timeout=5.0)
        return resp.status_code == 200
    except requests.RequestException:
        return False


# FastAPI app
app = FastAPI(title="Docstring Generation API", version="0.1.0")


class GenerateRequest(BaseModel):
    """Request model for docstring generation."""
    code: str
    max_new_tokens: Optional[int] = 256


class GenerateResponse(BaseModel):
    """Response model for docstring generation."""
    docstring: str


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a docstring for the given code snippet."""
    try:
        docstring = generate_docstring(request.code, request.max_new_tokens)
        return GenerateResponse(docstring=docstring)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint that verifies ollama is running."""
    is_healthy = check_ollama_health()
    if is_healthy:
        return {"status": "healthy", "service": "ollama"}
    else:
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy: ollama is not running or not accessible"
        )


def main():
    """Start the inference server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
