"""Tests for src.training.models module."""

import pytest

from src.training.models import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    ModelConfig,
    SamplingConfig,
    get_model_config,
    create_fallback_config,
    list_models,
)


class TestSamplingConfig:
    """Tests for SamplingConfig dataclass."""

    def test_default_values(self):
        """SamplingConfig should have sensible defaults."""
        config = SamplingConfig()
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.num_predict == 256

    def test_custom_values(self):
        """SamplingConfig should accept custom values."""
        config = SamplingConfig(temperature=1.0, top_p=0.95, top_k=50, num_predict=512)
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.num_predict == 512

    def test_immutable(self):
        """SamplingConfig should be immutable (frozen)."""
        config = SamplingConfig()
        with pytest.raises(AttributeError):
            config.temperature = 0.5


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_required_fields(self):
        """ModelConfig should require essential fields."""
        config = ModelConfig(
            name="Test Model",
            ollama_model="test:latest",
            context_window=4096,
            sampling=SamplingConfig()
        )
        assert config.name == "Test Model"
        assert config.ollama_model == "test:latest"
        assert config.context_window == 4096

    def test_default_values(self):
        """ModelConfig should have sensible defaults for optional fields."""
        config = ModelConfig(
            name="Test Model",
            ollama_model="test:latest",
            context_window=4096,
            sampling=SamplingConfig()
        )
        assert config.keep_alive == 0
        assert config.architecture == "dense"
        assert config.total_params is None
        assert config.memory_q4 is None
        assert config.description == ""

    def test_immutable(self):
        """ModelConfig should be immutable (frozen)."""
        config = ModelConfig(
            name="Test Model",
            ollama_model="test:latest",
            context_window=4096,
            sampling=SamplingConfig()
        )
        with pytest.raises(AttributeError):
            config.name = "New Name"


class TestModelRegistry:
    """Tests for MODEL_REGISTRY and related functions."""

    def test_registry_has_default_model(self):
        """Registry should contain the default model."""
        assert DEFAULT_MODEL_KEY in MODEL_REGISTRY

    def test_registry_models_are_valid(self):
        """All models in registry should be valid ModelConfig instances."""
        for key, config in MODEL_REGISTRY.items():
            assert isinstance(config, ModelConfig)
            assert isinstance(config.sampling, SamplingConfig)
            assert config.context_window > 0
            assert config.ollama_model

    def test_qwen25_coder_32b_config(self):
        """Qwen 2.5 Coder 32B should have correct configuration."""
        config = MODEL_REGISTRY["qwen2.5-coder-32b"]
        assert config.name == "Qwen 2.5 Coder 32B"
        assert config.ollama_model == "qwen2.5-coder:32b"
        assert config.context_window == 32768
        assert config.architecture == "dense"
        assert config.sampling.temperature == 0.7
        assert config.sampling.top_p == 0.9

    def test_qwen25_coder_7b_config(self):
        """Qwen 2.5 Coder 7B should have correct configuration."""
        config = MODEL_REGISTRY["qwen2.5-coder-7b"]
        assert config.name == "Qwen 2.5 Coder 7B"
        assert config.ollama_model == "qwen2.5-coder:7b"
        assert config.architecture == "dense"

    def test_qwen3_coder_30b_config(self):
        """Qwen3 Coder 30B should have correct MoE configuration."""
        config = MODEL_REGISTRY["qwen3-coder-30b"]
        assert config.name == "Qwen3 Coder 30B-A3B"
        assert config.ollama_model == "qwen3-coder:30b-a3b"
        assert config.context_window == 262144  # 256K
        assert config.architecture == "moe"
        # Qwen3 uses different sampling parameters
        assert config.sampling.temperature == 1.0
        assert config.sampling.top_p == 0.95
        # MoE models have longer keep_alive
        assert config.keep_alive == 300


class TestGetModelConfig:
    """Tests for get_model_config function."""

    def test_get_by_registry_key(self):
        """Should return config when given a valid registry key."""
        config = get_model_config("qwen2.5-coder-32b")
        assert config.ollama_model == "qwen2.5-coder:32b"

    def test_get_by_ollama_model_name(self):
        """Should return config when given a valid Ollama model name."""
        config = get_model_config("qwen2.5-coder:32b")
        assert config.name == "Qwen 2.5 Coder 32B"

    def test_fallback_for_unknown_model(self):
        """Should create fallback config for unknown models."""
        config = get_model_config("unknown-model:latest")
        assert config.ollama_model == "unknown-model:latest"
        assert config.architecture == "unknown"
        assert config.context_window == 32768  # Conservative default

    def test_fallback_uses_default_sampling(self):
        """Fallback config should use default sampling parameters."""
        config = get_model_config("custom-model:v1")
        assert config.sampling.temperature == 0.7
        assert config.sampling.top_p == 0.9
        assert config.sampling.top_k == 40


class TestCreateFallbackConfig:
    """Tests for create_fallback_config function."""

    def test_creates_valid_config(self):
        """Should create a valid ModelConfig for any model name."""
        config = create_fallback_config("my-model:latest")
        assert isinstance(config, ModelConfig)
        assert config.ollama_model == "my-model:latest"
        assert config.name == "my-model:latest"

    def test_uses_conservative_defaults(self):
        """Fallback should use conservative defaults."""
        config = create_fallback_config("test")
        assert config.context_window == 32768
        assert config.keep_alive == 0
        assert config.architecture == "unknown"


class TestListModels:
    """Tests for list_models function."""

    def test_returns_list(self):
        """Should return a list of model info dictionaries."""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) == len(MODEL_REGISTRY)

    def test_model_info_structure(self):
        """Each model info should have required fields."""
        models = list_models()
        for model_info in models:
            assert "key" in model_info
            assert "name" in model_info
            assert "ollama_model" in model_info
            assert "context_window" in model_info
            assert "architecture" in model_info
            assert "description" in model_info

    def test_contains_all_registry_models(self):
        """Should contain all models from the registry."""
        models = list_models()
        keys = {m["key"] for m in models}
        assert keys == set(MODEL_REGISTRY.keys())
