"""Model configuration registry for docstring generation.

Provides model-specific configurations including sampling parameters,
context windows, and memory requirements for different LLM backends.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters for text generation.

    Parameters
    ----------
    temperature : float
        Controls randomness in generation. Higher values = more random.
    top_p : float
        Nucleus sampling parameter. Considers tokens with cumulative probability >= top_p.
    top_k : int
        Limits sampling to top k most likely tokens.
    num_predict : int
        Maximum number of tokens to generate.
    """
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 256


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model.

    Parameters
    ----------
    name : str
        Human-readable display name for the model.
    ollama_model : str
        Ollama model identifier (e.g., "qwen2.5-coder:32b").
    context_window : int
        Maximum context length in tokens.
    sampling : SamplingConfig
        Default sampling parameters for this model.
    keep_alive : int
        Model keep-alive time in seconds (0 = unload immediately after request).
    architecture : str
        Model architecture type: "dense" or "moe".
    total_params : str, optional
        Total parameter count description (e.g., "32B", "30B (3.3B active)").
    memory_q4 : str, optional
        Approximate memory requirement at Q4 quantization.
    description : str
        Brief description of the model's characteristics.
    """
    name: str
    ollama_model: str
    context_window: int
    sampling: SamplingConfig
    keep_alive: int = 0
    architecture: str = "dense"
    total_params: Optional[str] = None
    memory_q4: Optional[str] = None
    description: str = ""


# Pre-defined model configurations
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen2.5-coder-32b": ModelConfig(
        name="Qwen 2.5 Coder 32B",
        ollama_model="qwen2.5-coder:32b",
        context_window=32768,
        sampling=SamplingConfig(temperature=0.7, top_p=0.9, top_k=40),
        keep_alive=0,
        architecture="dense",
        total_params="32B",
        memory_q4="~18GB",
        description="Dense 32B model, good balance of quality and speed"
    ),
    "qwen2.5-coder-14b": ModelConfig(
        name="Qwen 2.5 Coder 14B",
        ollama_model="qwen2.5-coder:14b",
        context_window=32768,
        sampling=SamplingConfig(temperature=0.7, top_p=0.9, top_k=40),
        keep_alive=0,
        architecture="dense",
        total_params="14B",
        memory_q4="~8GB",
        description="Mid-size dense model, balanced performance"
    ),
    "qwen2.5-coder-7b": ModelConfig(
        name="Qwen 2.5 Coder 7B",
        ollama_model="qwen2.5-coder:7b",
        context_window=32768,
        sampling=SamplingConfig(temperature=0.7, top_p=0.9, top_k=40),
        keep_alive=0,
        architecture="dense",
        total_params="7B",
        memory_q4="~4GB",
        description="Smaller variant, faster inference"
    ),
    "qwen3-coder-30b": ModelConfig(
        name="Qwen3 Coder 30B-A3B",
        ollama_model="qwen3-coder:30b-a3b",
        context_window=262144,  # 256K tokens
        sampling=SamplingConfig(temperature=1.0, top_p=0.95, top_k=40),
        keep_alive=300,  # Longer keep_alive for MoE model to avoid reload overhead
        architecture="moe",
        total_params="30B (3.3B active)",
        memory_q4="~18GB",
        description="MoE model with 256K context, best quality"
    ),
}

# Default model key
DEFAULT_MODEL_KEY = "qwen2.5-coder-32b"


def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a model by registry key or Ollama model name.

    Parameters
    ----------
    model_key : str
        Either a registry key (e.g., "qwen2.5-coder-32b") or a raw Ollama
        model name (e.g., "qwen2.5-coder:32b").

    Returns
    -------
    ModelConfig
        The model configuration. If the key is not found in the registry,
        creates a fallback configuration with default sampling parameters.
    """
    # First, try direct registry lookup
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]

    # Check if it matches any ollama_model in the registry
    for config in MODEL_REGISTRY.values():
        if config.ollama_model == model_key:
            return config

    # Fallback: create a default config for unknown models
    return create_fallback_config(model_key)


def create_fallback_config(ollama_model: str) -> ModelConfig:
    """Create a fallback configuration for unknown models.

    Parameters
    ----------
    ollama_model : str
        The Ollama model identifier.

    Returns
    -------
    ModelConfig
        A configuration with default sampling parameters.
    """
    return ModelConfig(
        name=ollama_model,
        ollama_model=ollama_model,
        context_window=32768,  # Conservative default
        sampling=SamplingConfig(),  # Use defaults
        keep_alive=0,
        architecture="unknown",
        description="Custom model (using default configuration)"
    )


def list_models() -> list[dict]:
    """List all available model configurations.

    Returns
    -------
    list of dict
        List of model information dictionaries containing key, name,
        ollama_model, context_window, architecture, memory_q4, and description.
    """
    return [
        {
            "key": key,
            "name": config.name,
            "ollama_model": config.ollama_model,
            "context_window": config.context_window,
            "architecture": config.architecture,
            "total_params": config.total_params,
            "memory_q4": config.memory_q4,
            "description": config.description,
        }
        for key, config in MODEL_REGISTRY.items()
    ]
