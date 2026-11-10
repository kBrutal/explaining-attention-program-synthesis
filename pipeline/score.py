"""Metric (Eq. 2) and the post-condition check every program output must pass."""
import numpy as np


def iou(A: np.ndarray, B: np.ndarray) -> float:
    """Σ min(A, B) / Σ max(A, B) for non-negative matrices of equal shape."""
    hi = np.maximum(A, B).sum()
    return float(np.minimum(A, B).sum() / hi) if hi > 0 else 1.0


def validate(output, expected_tokens: list[str], causal: bool) -> np.ndarray:
    """Check a program's (tokens, matrix) return value; return the float64 matrix or raise ValueError."""
    if not (isinstance(output, tuple) and len(output) == 2):
        raise ValueError("must return (tokens, attention_matrix)")
    tokens, A = output
    if list(tokens) != expected_tokens:
        raise ValueError("tokens differ from the model's tokenization")
    A = np.asarray(A, dtype=np.float64)
    n = len(expected_tokens)
    if A.shape != (n, n):
        raise ValueError(f"shape {A.shape}, expected {(n, n)}")
    if not np.isfinite(A).all() or (A < 0).any():
        raise ValueError("non-finite or negative entries")
    if causal and not np.allclose(np.triu(A, 1), 0, atol=1e-6):
        raise ValueError("attends to future tokens (j > i)")
    sums = A.sum(-1, keepdims=True)
    if not np.allclose(sums, 1, atol=1e-2):
        raise ValueError("rows must sum to 1")
    return A / sums  # absorb small float drift
