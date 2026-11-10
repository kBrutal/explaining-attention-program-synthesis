"""M10: LLM labels each library program with one of six categories; a head's category is that of its
best-fit program (Fig. 4b). Also the A.5 sufficiency check: max IoU between category-average patterns.

Writes runs/<model>/categories.json. Run: python -m pipeline.categorize gpt2
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from pipeline import llm
from pipeline.crossmatch import best_programs, library
from pipeline.extract import run_dir
from pipeline.sandbox import run_program
from pipeline.summarize import MODEL_TEXT

CATEGORIES = ["First Token", "Coreference", "Final Token", "Syntactic", "Sequential", "Uniform"]
TEMPLATE = Path("prompts/categorize.txt")
SUFFICIENCY_SPLIT = "stories"


def label(model_name: str, src: str) -> str | None:
    prompt = TEMPLATE.read_text().format(model_display=MODEL_TEXT[model_name]["display"], src=src)
    reply = llm.complete([{"role": "user", "content": prompt}])["reply"]
    m = re.findall(r"CATEGORY:\s*\**\s*([A-Za-z ]+?)\s*\**\s*$", reply, re.M)
    return next((c for c in CATEGORIES if m and m[-1].lower() == c.lower()), None)


def sufficiency(model_name: str, sources: list[str], program_cats: list[str | None]) -> dict:
    """Mean predicted pattern per category on each input, then mean IoU between every pair of categories."""
    sums, counts = {}, {}  # (category, input index) → summed matrix / number of programs
    for src, c in zip(sources, program_cats):
        for k, P in enumerate(run_program(model_name, src, SUFFICIENCY_SPLIT)[0]):
            if c and P is not None:
                sums[c, k] = sums.get((c, k), 0) + P
                counts[c, k] = counts.get((c, k), 0) + 1
    cats = sorted({c for c, _ in sums})
    pairs = {}
    for a, b in combinations(cats, 2):
        ks = [k for (c, k) in sums if c == a and (b, k) in sums]
        pairs[f"{a} vs {b}"] = float(np.mean([
            np.minimum(sums[a, k] / counts[a, k], sums[b, k] / counts[b, k]).sum()
            / np.maximum(sums[a, k] / counts[a, k], sums[b, k] / counts[b, k]).sum() for k in ks]))
    return pairs


def categorize(model_name: str) -> dict:
    out = run_dir(model_name) / "categories.json"
    if out.exists():
        return json.loads(out.read_text())
    records = library(model_name)
    sources = [r["candidates"][r["chosen"]]["src"] for r in records]
    with ThreadPoolExecutor(16) as pool:
        program_cats = list(pool.map(lambda s: label(model_name, s), sources))
    best, _ = best_programs(model_name)
    result = {
        "programs": {f"L{r['layer']}H{r['head']}": c for r, c in zip(records, program_cats)},
        "heads": [[program_cats[p] for p in row] for row in best.tolist()],  # [L][H] via best-fit program
        "sufficiency": sufficiency(model_name, sources, program_cats),
    }
    out.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":  # required: sandbox children re-import the main module
    for m in sys.argv[1:]:
        r = categorize(m)
        progs, heads = list(r["programs"].values()), [c for row in r["heads"] for c in row]
        print(f"{m}: unlabelled programs {progs.count(None)}")
        for c in CATEGORIES:
            per_layer = [row.count(c) for row in r["heads"]]
            print(f"  {c:12s} programs {progs.count(c):3d} | heads {heads.count(c):3d} | heads per layer {per_layer}")
        worst = max(r["sufficiency"].items(), key=lambda kv: kv[1])
        print(f"  sufficiency: max cross-category IoU {worst[1]:.3f} ({worst[0]}) → "
              f"{'passes' if worst[1] < 0.4 else 'fails'} the 0.4 threshold (paper: fails for every model)")
