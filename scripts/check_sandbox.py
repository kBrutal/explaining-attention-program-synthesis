"""M5 check: sandbox handles good/bad programs, caching works, scores match the M3 reference."""
import shutil
import time

import numpy as np

from pipeline.extract import load_attention, run_dir
from pipeline.sandbox import extract_code, program_id, run_program, score_program

MODEL, SPLIT = "gpt2", "val"

PREV = """
def predict_attention_map(sentence):
    tokens = gpt2_tokenize(sentence)
    A = np.eye(len(tokens), k=-1)
    A[0, 0] = 1
    return tokens, apply_causal_mask(A)
"""
BAD = {
    "syntax": "def predict_attention_map(sentence) return 1",
    "missing_fn": "def f(s): pass",
    "runtime": "def predict_attention_map(sentence):\n    return 1 / 0",
    "hang": "def predict_attention_map(sentence):\n    while True: pass",
    "network": "import urllib.request\ndef predict_attention_map(sentence):\n"
               "    urllib.request.urlopen('http://example.com')",
    "future": "def predict_attention_map(sentence):\n    t = gpt2_tokenize(sentence)\n"
              "    return t, np.ones((len(t), len(t))) / len(t)",
    "crash": "import os\ndef predict_attention_map(sentence):\n    os._exit(1)",
}


def main():
    assert extract_code("text\n```python\ndef f():\n    pass\n```\nmore") == "def f():\n    pass"
    assert extract_code("no code here") is None

    for src in [PREV, *BAD.values()]:  # start from a clean cache
        shutil.rmtree(run_dir(MODEL) / "outputs" / program_id(MODEL, src), ignore_errors=True)

    attn = load_attention(MODEL, SPLIT)
    t0 = time.time()
    scores, errors = score_program(MODEL, PREV, SPLIT, attn)
    cold = time.time() - t0
    assert all(e is None for e in errors), errors[:3]
    assert scores[4, 11] > 0.8, scores[4, 11]  # L4H11 = previous-token head (M3)
    t0 = time.time()
    cached, _ = score_program(MODEL, PREV, SPLIT, attn)
    warm = time.time() - t0
    assert np.allclose(scores, cached, atol=1e-3)  # cache stores float16
    print(f"previous_token: L4H11 IoU={scores[4, 11]:.3f}, cold {cold:.1f}s, cached {warm:.2f}s")

    for name, src in BAD.items():
        mats, errors = run_program(MODEL, src, SPLIT)
        assert all(m is None for m in mats) and all(errors), name
        s, _ = score_program(MODEL, src, SPLIT, attn)
        assert (s == 0).all(), name
        print(f"{name:10s} → score 0, e.g. {errors[0][:70]!r}")

    print("OK")


if __name__ == "__main__":  # required: spawned sandbox children re-import this module
    main()
