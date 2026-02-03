"""Tests for src.training.serve module."""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.training.serve import app, check_ollama_health, generate_docstring, get_active_model
from src.training.models import MODEL_REGISTRY, DEFAULT_MODEL_KEY, get_model_config


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @patch("src.training.serve.requests.get")
    def test_health_success(self, mock_get, client):
        """Health check should return 200 when ollama is running."""
        # Mock successful response from ollama
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ollama"
        # New fields for model info
        assert "active_model" in data
        assert "ollama_model" in data
        mock_get.assert_called_once()

    @patch("src.training.serve.requests.get")
    def test_health_failure_connection_error(self, mock_get, client):
        """Health check should return 503 when ollama is not accessible."""
        # Mock connection error
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")

        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert "unhealthy" in data["detail"].lower()
        assert "ollama" in data["detail"].lower()

    @patch("src.training.serve.requests.get")
    def test_health_failure_non_200_status(self, mock_get, client):
        """Health check should return 503 when ollama returns non-200 status."""
        # Mock non-200 response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        response = client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert "unhealthy" in data["detail"].lower()

    @patch("src.training.serve.requests.get")
    def test_health_failure_timeout(self, mock_get, client):
        """Health check should return 503 when ollama request times out."""
        import requests
        mock_get.side_effect = requests.Timeout("Request timed out")

        response = client.get("/health")

        assert response.status_code == 503


class TestGenerateEndpoint:
    """Tests for the /generate endpoint."""

    @patch("src.training.serve.requests.post")
    def test_generate_success_message_format(self, mock_post, client):
        """Generate should return docstring when ollama responds with message format."""
        # Mock successful ollama response with message format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": '"""Compute the sum of two numbers.\n\nParameters\n----------\nx : int\n    First number.\ny : int\n    Second number.\n\nReturns\n-------\nint\n    Sum of x and y.\n"""'
            }
        }
        mock_post.return_value = mock_response

        request_data = {
            "code": "def add(x, y):\n    return x + y",
            "max_new_tokens": 256
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "docstring" in data
        assert "Compute the sum" in data["docstring"]
        # New field for model info
        assert "model" in data
        mock_post.assert_called_once()

    @patch("src.training.serve.requests.post")
    def test_generate_success_response_format(self, mock_post, client):
        """Generate should return docstring when ollama responds with response format."""
        # Mock successful ollama response with response format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '"""Return the product of two numbers.\n\nParameters\n----------\na : float\n    First number.\nb : float\n    Second number.\n\nReturns\n-------\nfloat\n    Product of a and b.\n"""'
        }
        mock_post.return_value = mock_response

        request_data = {
            "code": "def multiply(a, b):\n    return a * b"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "docstring" in data
        assert "Return the product" in data["docstring"]

    @patch("src.training.serve.requests.post")
    def test_generate_success_choices_format(self, mock_post, client):
        """Generate should return docstring when ollama responds with choices format."""
        # Mock successful ollama response with choices format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '"""Calculate the difference.\n\nParameters\n----------\nx : int\n    First number.\ny : int\n    Second number.\n\nReturns\n-------\nint\n    Difference of x and y.\n"""'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        request_data = {
            "code": "def subtract(x, y):\n    return x - y",
            "max_new_tokens": 128
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "docstring" in data
        assert "Calculate the difference" in data["docstring"]

    @patch("src.training.serve.requests.post")
    def test_generate_default_max_new_tokens(self, mock_post, client):
        """Generate should use default max_new_tokens when not provided."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '"""Test docstring."""'}
        }
        mock_post.return_value = mock_response

        request_data = {"code": "def test(): pass"}
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200
        # Verify that max_new_tokens was included in the payload
        call_args = mock_post.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert "options" in payload
        assert payload["options"]["num_predict"] == 256

    @patch("src.training.serve.requests.post")
    def test_generate_failure_connection_error(self, mock_post, client):
        """Generate should return 500 when ollama connection fails."""
        import requests
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        request_data = {
            "code": "def test(): pass"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to generate docstring" in data["detail"]

    @patch("src.training.serve.requests.post")
    def test_generate_failure_timeout(self, mock_post, client):
        """Generate should return 500 when ollama request times out."""
        import requests
        mock_post.side_effect = requests.Timeout("Request timed out")

        request_data = {
            "code": "def test(): pass"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    @patch("src.training.serve.requests.post")
    def test_generate_failure_non_200_status(self, mock_post, client):
        """Generate should return 500 when ollama returns non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Internal server error")
        mock_post.return_value = mock_response

        request_data = {
            "code": "def test(): pass"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 500

    @patch("src.training.serve.requests.post")
    def test_generate_failure_unexpected_format(self, mock_post, client):
        """Generate should return 500 when ollama returns unexpected format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "unexpected": "format"
        }
        mock_post.return_value = mock_response

        request_data = {
            "code": "def test(): pass"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Unexpected response format" in data["detail"]

    def test_generate_missing_code_field(self, client):
        """Generate should return 422 when code field is missing."""
        request_data = {}
        response = client.post("/generate", json=request_data)

        assert response.status_code == 422

    def test_generate_empty_code(self, client):
        """Generate should accept empty code string."""
        with patch("src.training.serve.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {"content": '"""Empty function."""'}
            }
            mock_post.return_value = mock_response

            request_data = {"code": ""}
            response = client.post("/generate", json=request_data)

            assert response.status_code == 200


class TestHelperFunctions:
    """Tests for helper functions."""

    @patch("src.training.serve.requests.get")
    def test_check_ollama_health_success(self, mock_get):
        """check_ollama_health should return True when ollama is accessible."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = check_ollama_health()

        assert result is True
        mock_get.assert_called_once()

    @patch("src.training.serve.requests.get")
    def test_check_ollama_health_failure(self, mock_get):
        """check_ollama_health should return False when ollama is not accessible."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")

        result = check_ollama_health()

        assert result is False

    @patch("src.training.serve.requests.post")
    def test_generate_docstring_success(self, mock_post):
        """generate_docstring should return docstring content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '"""Test docstring."""'}
        }
        mock_post.return_value = mock_response

        result = generate_docstring("def test(): pass", max_new_tokens=128)

        assert result == '"""Test docstring."""'
        mock_post.assert_called_once()

    @patch("src.training.serve.requests.post")
    def test_generate_docstring_failure(self, mock_post):
        """generate_docstring should raise RuntimeError on failure."""
        import requests
        mock_post.side_effect = requests.RequestException("Connection error")

        with pytest.raises(RuntimeError) as exc_info:
            generate_docstring("def test(): pass")

        assert "Failed to generate docstring" in str(exc_info.value)


class TestModelsEndpoint:
    """Tests for the /models endpoint."""

    def test_list_models_returns_all(self, client):
        """GET /models should return all registered models."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) == len(MODEL_REGISTRY)

    def test_list_models_includes_default(self, client):
        """Response should indicate the default model."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "default" in data
        assert data["default"] == DEFAULT_MODEL_KEY

    def test_list_models_includes_active(self, client):
        """Response should indicate the active model."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert "active" in data

    def test_list_models_model_structure(self, client):
        """Each model in the list should have required fields."""
        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        for model in data["models"]:
            assert "key" in model
            assert "name" in model
            assert "ollama_model" in model
            assert "context_window" in model
            assert "architecture" in model


class TestModelSelection:
    """Tests for per-request model selection."""

    @patch("src.training.serve.requests.post")
    def test_generate_with_specific_model_key(self, mock_post, client):
        """Should use the specified model key in the request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '"""Test docstring."""'}
        }
        mock_post.return_value = mock_response

        request_data = {
            "code": "def test(): pass",
            "model": "qwen2.5-coder-7b"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "qwen2.5-coder:7b"

        # Verify the correct model was used in the payload
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "qwen2.5-coder:7b"

    @patch("src.training.serve.requests.post")
    def test_generate_with_raw_ollama_model(self, mock_post, client):
        """Should accept raw Ollama model names."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '"""Test docstring."""'}
        }
        mock_post.return_value = mock_response

        request_data = {
            "code": "def test(): pass",
            "model": "qwen2.5-coder:14b"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200

    @patch("src.training.serve.requests.post")
    def test_generate_applies_model_sampling(self, mock_post, client):
        """Should apply model-specific sampling parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '"""Test docstring."""'}
        }
        mock_post.return_value = mock_response

        # Use Qwen3 which has different sampling params
        request_data = {
            "code": "def test(): pass",
            "model": "qwen3-coder-30b"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200

        # Verify Qwen3 sampling parameters were used
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        options = payload["options"]
        assert options["temperature"] == 1.0
        assert options["top_p"] == 0.95
        assert options["top_k"] == 40

    @patch("src.training.serve.requests.post")
    def test_generate_applies_model_keep_alive(self, mock_post, client):
        """Should apply model-specific keep_alive setting."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '"""Test docstring."""'}
        }
        mock_post.return_value = mock_response

        # Use Qwen3 which has keep_alive=300
        request_data = {
            "code": "def test(): pass",
            "model": "qwen3-coder-30b"
        }
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200

        # Verify Qwen3 keep_alive was used
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["keep_alive"] == 300

    @patch("src.training.serve.requests.post")
    def test_generate_without_model_uses_default(self, mock_post, client):
        """Should use default model when none specified."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '"""Test docstring."""'}
        }
        mock_post.return_value = mock_response

        request_data = {"code": "def test(): pass"}
        response = client.post("/generate", json=request_data)

        assert response.status_code == 200

        # Verify default model was used
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        default_config = get_model_config(DEFAULT_MODEL_KEY)
        assert payload["model"] == default_config.ollama_model


class TestHealthWithModel:
    """Tests for health endpoint with model information."""

    @patch("src.training.serve.requests.get")
    def test_health_reports_active_model(self, mock_get, client):
        """Health endpoint should report the active model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "active_model" in data
        assert "ollama_model" in data
        # Should match the configured model
        active_model = get_active_model()
        assert data["active_model"] == active_model.name
        assert data["ollama_model"] == active_model.ollama_model
