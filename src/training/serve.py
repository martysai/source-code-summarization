"""FastAPI inference server for docstring generation using multiple LLM backends.

Serves predictions via HTTP using Ollama as the backend. Supports multiple models
including Qwen 2.5 Coder and Qwen3 Coder variants with model-specific configurations.

Usage:
    uvicorn src.training.serve:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /generate  - Generate a docstring for a given code snippet
    GET  /health    - Health check
    GET  /models    - List available models
"""

import os
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.training.models import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    ModelConfig,
    get_model_config,
    list_models,
)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL_KEY)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120.0"))


def get_active_model() -> ModelConfig:
    """Get the currently active model configuration from environment."""
    return get_model_config(OLLAMA_MODEL)


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


def generate_docstring(
    code: str,
    max_new_tokens: Optional[int] = None,
    model_config: Optional[ModelConfig] = None
) -> str:
    """Generate a docstring for the given code snippet using ollama API.

    Parameters
    ----------
    code : str
        The Python code snippet to generate a docstring for.
    max_new_tokens : int, optional
        Maximum number of tokens to generate. If None, uses the model's default.
    model_config : ModelConfig, optional
        Model configuration to use. If None, uses the active model from environment.

    Returns
    -------
    str
        The generated docstring.

    Raises
    ------
    RuntimeError
        If the request to ollama fails.
    ValueError
        If the response format is unexpected.
    """
    if model_config is None:
        model_config = get_active_model()

    # Use model-specific defaults if not provided
    if max_new_tokens is None:
        max_new_tokens = model_config.sampling.num_predict

    payload = {
        "model": model_config.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": code}
        ],
        "stream": False,
        "keep_alive": model_config.keep_alive,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": model_config.sampling.temperature,
            "top_p": model_config.sampling.top_p,
            "top_k": model_config.sampling.top_k,
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
app = FastAPI(
    title="Docstring Generation API",
    version="0.2.0",
    description="Generate Python docstrings using Qwen Coder models via Ollama"
)


class GenerateRequest(BaseModel):
    """Request model for docstring generation."""
    code: str
    max_new_tokens: Optional[int] = None
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response model for docstring generation."""
    docstring: str
    model: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    service: str
    active_model: str
    ollama_model: str


class ModelsResponse(BaseModel):
    """Response model for listing available models."""
    default: str
    active: str
    models: list


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a docstring for the given code snippet.

    Optionally specify a model to use for generation. If not specified,
    uses the active model from the OLLAMA_MODEL environment variable.
    """
    try:
        # Determine which model to use
        if request.model:
            model_config = get_model_config(request.model)
        else:
            model_config = get_active_model()

        docstring = generate_docstring(
            request.code,
            request.max_new_tokens,
            model_config
        )
        return GenerateResponse(
            docstring=docstring,
            model=model_config.ollama_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint that verifies ollama is running.

    Returns information about the active model configuration.
    """
    is_healthy = check_ollama_health()
    active_model = get_active_model()

    if is_healthy:
        return HealthResponse(
            status="healthy",
            service="ollama",
            active_model=active_model.name,
            ollama_model=active_model.ollama_model
        )
    else:
        raise HTTPException(
            status_code=503,
            detail="Service unhealthy: ollama is not running or not accessible"
        )


@app.get("/models", response_model=ModelsResponse)
async def get_models():
    """List all available model configurations.

    Returns the default model, currently active model, and a list of
    all registered models with their configurations.
    """
    active_model = get_active_model()
    return ModelsResponse(
        default=DEFAULT_MODEL_KEY,
        active=OLLAMA_MODEL,
        models=list_models()
    )


def main():
    """Start the inference server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
