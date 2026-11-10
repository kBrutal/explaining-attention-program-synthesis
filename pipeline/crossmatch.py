"""M8: score every program in a model's library against every head → best program per head (Table A.6).

Library = the intended (chosen) program of each head. val/test outputs are cached from M7; the
whole-story split runs each program once more. Writes runs/<model>/library.npz:
  val, test, stories: [P, L, H] mean IoU of program p against head (l, h); program p belongs to head heads[p].
"Best" is picked on val for sentence splits (no test-set selection) and on the split itself for
stories (the only story split; paper-style, slightly optimistic).

Run: python -m pipeline.crossmatch gpt2
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pipeline.extract import load_attention, run_dir
from pipeline.sandbox import head_ious
from pipeline.synthesize import sandboxed

SPLITS = ("val", "test", "stories")


def library(model_name: str) -> list[dict]:
    """Head records sorted by (layer, head)."""
    records = [json.loads(p.read_text()) for p in (run_dir(model_name) / "programs").glob("L*H*.json")]
    return sorted(records, key=lambda r: (r["layer"], r["head"]))


def crossmatch(model_name: str) -> dict:
    out = run_dir(model_name) / "library.npz"
    if out.exists():
        with np.load(out) as z:
            return dict(z)
    records = library(model_name)
    scores = {}
    for split in SPLITS:
        attn = load_attention(model_name, split)

        def score(r, split=split, attn=attn):  # cached outputs return instantly; new ones run 4 at a time
            return head_ious(sandboxed(model_name, r["candidates"][r["chosen"]]["src"], split)[0], attn)
        with ThreadPoolExecutor(8) as pool:
            scores[split] = np.stack(list(pool.map(score, records)))
        print(f"  {model_name} {split}: scored {len(records)} programs", flush=True)
    scores["heads"] = np.array([(r["layer"], r["head"]) for r in records])
    np.savez(out, **scores)
    return scores


def best_programs(model_name: str, split: str = "stories") -> tuple[np.ndarray, np.ndarray]:
    """(program index [L, H], its IoU [L, H]) of each head's best-fit program, picked and scored on `split`."""
    scores = crossmatch(model_name)[split]
    return scores.argmax(0), scores.max(0)


def report(model_name: str) -> dict:
    s = crossmatch(model_name)
    heads, (P, L, H) = s["heads"], s["test"].shape
    result = {}
    for split, pick in (("test", "val"), ("stories", "stories")):
        scores, picker = s[split].reshape(P, -1), s[pick].reshape(P, -1)
        own = scores[np.arange(P), heads[:, 0] * H + heads[:, 1]]  # intended IoU per head
        best = scores[picker.argmax(0), np.arange(L * H)]
        differs = np.mean(picker.argmax(0)[heads[:, 0] * H + heads[:, 1]] != np.arange(P))
        r = result[split] = {"intended": float(own.mean()), "intended_median": float(np.median(own)),
                             "best": float(best.mean()), "best_median": float(np.median(best)),
                             "best_differs": float(differs)}
        print(f"{model_name} [{split}, best picked on {pick}]: intended IoU mean {r['intended']:.3f} "
              f"(median {r['intended_median']:.3f}) → best {r['best']:.3f} (median {r['best_median']:.3f}) | "
              f"best ≠ intended on {differs:.1%} of heads")
    return result


if __name__ == "__main__":  # required: sandbox children re-import the main module
    for m in sys.argv[1:]:
        report(m)
