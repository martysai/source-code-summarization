"""Fine-tune a code LLM with LoRA for docstring generation.

Supports small code models (1-3B parameters) fine-tuned with PEFT/LoRA
on the prepared instruction-tuning dataset. Designed for 1-2 A100 GPUs
with optional 4-bit quantization (QLoRA).

Usage:
    python -m src.training.train_lora \
        --model-name Salesforce/codegen2-1B \
        --dataset-path data/processed/python-method \
        --output-dir outputs/codegen2-1b-lora
"""


def load_model_and_tokenizer(model_name: str, quantize: bool = True):
    """Load base model with optional 4-bit quantization via bitsandbytes."""
    raise NotImplementedError


def create_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Create a LoRA configuration targeting attention projections."""
    raise NotImplementedError


def format_example(example: dict, tokenizer) -> dict:
    """Format a single example as instruction-response for the model."""
    raise NotImplementedError


def train(model_name: str, dataset_path: str, output_dir: str, **kwargs):
    """Run the full LoRA fine-tuning pipeline."""
    raise NotImplementedError


def main():
    """Entry point for LoRA training."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
