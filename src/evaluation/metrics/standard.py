"""Standard text generation metrics for code summarization evaluation.

Wraps BLEU and ROUGE-L from the HuggingFace evaluate library for consistent
use within the benchmark framework.
"""


def compute_bleu(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute BLEU score between predictions and references."""
    raise NotImplementedError


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-L score between predictions and references."""
    raise NotImplementedError
