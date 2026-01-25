"""FastAPI inference server for the fine-tuned docstring generation model.

Loads a LoRA-adapted model and serves predictions via HTTP. Designed to be
called by the VS Code extension for local, offline docstring generation.

Usage:
    uvicorn src.training.serve:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /generate  - Generate a docstring for a given code snippet
    GET  /health    - Health check
"""


def load_model(base_model: str, adapter_path: str):
    """Load base model + LoRA adapter for inference."""
    raise NotImplementedError


def generate_docstring(code: str, max_new_tokens: int = 256) -> str:
    """Generate a docstring for the given code snippet."""
    raise NotImplementedError


# FastAPI app will be defined here once dependencies are implemented.
# app = FastAPI()
#
# @app.post("/generate")
# async def generate(request: dict): ...
#
# @app.get("/health")
# async def health(): ...


def main():
    """Start the inference server."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
