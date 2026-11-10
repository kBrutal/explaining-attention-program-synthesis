"""M12: perplexity as more heads are replaced (Fig. 5a, 5b). GPT-2 only (BERT isn't a causal LM).

Heads are replaced greedily in descending best-fit IoU (stories split). Conditions:
  chosen    each head → its best-fit program
  wrong_s#  each head → a random program from a random *other* category (seeds 0, 1, 2)
  prune     each head's output zeroed
Evaluated on the ppl_eval stories (disjoint from the `stories` split used to pick best programs).
Writes runs/<model>/ppl.json.

Run: python -m pipeline.perplexity gpt2
"""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from pipeline.categorize import categorize
from pipeline.crossmatch import best_programs, library
from pipeline.extract import run_dir
from pipeline.replace import load_lm, nll, replaced
from pipeline.sandbox import program_id
from pipeline.synthesize import sandboxed

SPLIT = "ppl_eval"
STEP = 2  # heads added per level: 73 levels for 144 heads (~1.4% steps, like Fig. 5a)
SEEDS = (0, 1, 2)


def wrong_category_assignment(head_cats, program_cats, seed) -> dict[tuple[int, int], int | None]:
    """head → random program from a uniformly random category other than the head's own (paper A.5)."""
    rng = np.random.default_rng(seed)
    by_cat = {}
    for p, c in enumerate(program_cats):
        by_cat.setdefault(c, []).append(p)
    out = {}
    for l, row in enumerate(head_cats):
        for h, own in enumerate(row):
            other = sorted(c for c in by_cat if c != own and c is not None)
            out[l, h] = int(rng.choice(by_cat[other[rng.integers(len(other))]]))
    return out


def prepare(model_name: str, split: str):
    """Shared by M12 and M13: run every library program on `split` (cached), then return
    (head order by best IoU, best IoU [L, H], {condition: {head: program index or None}}, matrix(p, k, n))."""
    records = library(model_name)
    sources = [r["candidates"][r["chosen"]]["src"] for r in records]
    with ThreadPoolExecutor(8) as pool:
        failures = sum(sum(e is not None for e in errs)
                       for _, errs in pool.map(lambda s: sandboxed(model_name, s, split), sources))
    print(f"programs on {split}: {failures} failed (program, input) pairs → uniform-causal fallback", flush=True)

    best, best_iou = best_programs(model_name)
    L, H = best.shape
    order = sorted(((l, h) for l in range(L) for h in range(H)), key=lambda lh: -best_iou[lh])
    cats = categorize(model_name)
    program_cats = [cats["programs"][f"L{r['layer']}H{r['head']}"] for r in records]
    assign: dict[str, dict[tuple[int, int], int | None]] = {"chosen": {lh: int(best[lh]) for lh in order}}
    assign |= {f"wrong_s{s}": wrong_category_assignment(cats["heads"], program_cats, s) for s in SEEDS}
    assign["prune"] = {lh: None for lh in order}

    out_dirs = [run_dir(model_name) / "outputs" / program_id(model_name, s) for s in sources]
    errors = [json.loads((d / f"{split}_errors.json").read_text()) for d in out_dirs]
    npz = [np.load(d / f"{split}.npz") for d in out_dirs]

    def matrix(p: int | None, k: int, n: int) -> torch.Tensor | None:
        """Program p's prediction on input k; None = prune; a failed run falls back to uniform causal."""
        if p is None:
            return None
        if errors[p][k] is not None:
            return torch.tril(torch.ones(n, n)) / torch.arange(1, n + 1)[:, None]
        P = torch.from_numpy(npz[p][str(k)].astype(np.float32))
        return P / P.sum(-1, keepdim=True)  # undo float16 rounding

    return order, best_iou, assign, matrix, failures


def sweep(model_name: str = "gpt2") -> dict:
    out_path = run_dir(model_name) / "ppl.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    order, best_iou, assign, matrix, failures = prepare(model_name, SPLIT)
    L, H = best_iou.shape
    levels = list(range(0, L * H + 1, STEP))
    if levels[-1] != L * H:
        levels.append(L * H)
    stories = json.loads(open("data/splits.json").read())[SPLIT]
    tok, model = load_lm(model_name)
    totals = {c: np.zeros(len(levels)) for c in assign}
    n_tokens, t0 = 0, time.time()
    for k, story in enumerate(stories):
        ids = tok(story, return_tensors="pt")["input_ids"]
        n = ids.shape[1]
        base, count = nll(model, ids)
        n_tokens += count
        for cond, a in assign.items():
            totals[cond][0] += base
            mats = {lh: matrix(a[lh], k, n) for lh in order}
            for i, level in enumerate(levels[1:], 1):
                with replaced({lh: mats[lh] for lh in order[:level]}):
                    totals[cond][i] += nll(model, ids)[0]
        print(f"[{k + 1}/{len(stories)} {time.time() - t0:5.0f}s]", flush=True)

    ppl = {c: np.exp(t / n_tokens) for c, t in totals.items()}
    result = {"levels": levels, "fraction": [lv / (L * H) for lv in levels], "n_stories": len(stories),
              "n_tokens": n_tokens, "failed_program_story_pairs": int(failures),
              "ppl": {c: v.tolist() for c, v in ppl.items()},
              "increase_pct": {c: (100 * (v / v[0] - 1)).tolist() for c, v in ppl.items()},
              "mean_iou_replaced": [float(np.mean([best_iou[lh] for lh in order[:lv]])) if lv else None
                                    for lv in levels]}
    # Fig. 5b: mean IoU of the replaced set vs log ppl increase (chosen condition, levels with an increase)
    xs, ys = zip(*[(x, math.log(y)) for x, y in zip(result["mean_iou_replaced"][1:], result["increase_pct"]["chosen"][1:])
                   if y > 0])
    result["fig5b_pearson_r"] = float(np.corrcoef(xs, ys)[0, 1])
    out_path.write_text(json.dumps(result, indent=1))
    return result


def at(result, frac):
    i = int(np.argmin([abs(f - frac) for f in result["fraction"]]))
    return result["fraction"][i], {c: round(v[i], 1) for c, v in result["increase_pct"].items()}


if __name__ == "__main__":  # required: sandbox children re-import the main module
    r = sweep(sys.argv[1] if len(sys.argv) > 1 else "gpt2")
    print(f"base ppl {r['ppl']['chosen'][0]:.2f} over {r['n_stories']} stories / {r['n_tokens']} tokens")
    for frac in (0.1, 0.25, 0.5, 1.0):
        f, inc = at(r, frac)
        print(f"  {f:.0%} heads replaced: ppl increase % {inc}")
    print(f"Fig. 5b Pearson r (mean IoU vs log ppl increase) {r['fig5b_pearson_r']:.3f} (paper GPT-2: -0.915)")
