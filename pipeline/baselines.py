"""M9: baseline attention patterns, scored like programs (Fig. 3).

Each baseline: (n, causal, rng) → n×n row-stochastic matrix. Writes runs/<model>/baselines.npz
with a [L, H] mean IoU per "<split>/<baseline>" for split in test, stories.

Run: python -m pipeline.baselines gpt2
"""
import sys

import numpy as np

from pipeline.extract import MODELS, load_attention, run_dir
from pipeline.sandbox import head_ious

SEED = 0


def allowed(n: int, causal: bool) -> np.ndarray:
    return np.tril(np.ones((n, n))) if causal else np.ones((n, n))


def random_token(n, causal, rng):
    """One-hot per row on a random allowed key."""
    keys = [rng.integers(i + 1 if causal else n) for i in range(n)]
    return np.eye(n)[keys]


def true_random(n, causal, rng):
    """Random row-stochastic matrix: Dirichlet(1) over each row's allowed keys."""
    A = rng.dirichlet(np.ones(n), size=n) * allowed(n, causal)
    return A / A.sum(1, keepdims=True)


def random_column(n, causal, rng):
    """All rows on one random column; rows that can't see it (causal, i < c) are uniform over j ≤ i."""
    A = np.zeros((n, n))
    A[:, rng.integers(n)] = 1
    A *= allowed(n, causal)
    empty = A.sum(1) == 0
    A[empty] = allowed(n, causal)[empty]
    return A / A.sum(1, keepdims=True)


def uniform(n, causal, rng):
    """Lower-diagonal uniform for decoders, fully uniform for BERT."""
    A = allowed(n, causal)
    return A / A.sum(1, keepdims=True)


def first_token(n, causal, rng):
    """Extra (not in the paper): everything on token 0. Shows how much of the IoU is attention sinks."""
    A = np.zeros((n, n))
    A[:, 0] = 1
    return A


BASELINES = {"Random Token": random_token, "True Random": true_random, "Random Column": random_column,
             "Uniform": uniform, "First Token (extra)": first_token}


def baselines(model_name: str) -> dict:
    out = run_dir(model_name) / "baselines.npz"
    if out.exists():
        with np.load(out) as z:
            return dict(z)
    causal, scores = MODELS[model_name]["causal"], {}
    for split in ("test", "stories"):
        attn = load_attention(model_name, split)
        for name, fn in BASELINES.items():
            rng = np.random.default_rng(SEED)
            scores[f"{split}/{name}"] = head_ious([fn(len(tokens), causal, rng) for tokens, _ in attn], attn)
    np.savez(out, **scores)
    return scores


if __name__ == "__main__":
    for m in sys.argv[1:]:
        s = baselines(m)
        for split in ("test", "stories"):
            print(f"{m} [{split}]: " + " | ".join(f"{k.split('/')[1]} {v.mean():.3f} (median {np.median(v):.3f})"
                                                for k, v in s.items() if k.startswith(split)))
