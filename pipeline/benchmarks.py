"""M13: zero-shot multiple-choice accuracy as heads are replaced (Fig. 6, Table 1). GPT-2 only.

Prompts follow lm-eval-harness: score log P(" " + choice | context) for each choice; acc = argmax of the
summed log-likelihood, acc_norm = argmax of log-likelihood per character of the choice.
200 examples per benchmark (paper A.5), seeded sample. Each context + choice string is one input
text (split "bench"), so programs predict attention on exactly the token sequence the model sees.

  python -m pipeline.benchmarks build        → data/bench.json + splits.json["bench"] + tokens
  python -m pipeline.benchmarks gpt2         → runs/gpt2/bench.json
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset

from pipeline.extract import MODELS, SPLITS_FILE, add_token_splits, run_dir
from pipeline.perplexity import prepare
from pipeline.replace import load_lm, replaced

N_EXAMPLES, SEED = 200, 0
LEVELS = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]  # 10 levels in 5–100% + 0
BENCH_FILE = Path("data/bench.json")


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def _copa(d):
    connector = {"cause": "because", "effect": "therefore"}[d["question"]]
    return (d["premise"].strip()[:-1] + f" {connector}",
            [c[0].lower() + c[1:] for c in (d["choice1"], d["choice2"])], d["label"])


# name → (dataset loader, doc → (context, choices, gold index))
BENCHMARKS = {
    "hellaswag": (lambda: load_dataset("Rowan/hellaswag", split="validation"),
                  lambda d: (_hellaswag_clean(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()),
                             [_hellaswag_clean(e) for e in d["endings"]], int(d["label"]))),
    "piqa": (lambda: load_dataset("baber/piqa", split="validation"),
             lambda d: (f"Question: {d['goal']}\nAnswer:", [d["sol1"], d["sol2"]], d["label"])),
    "sciq": (lambda: load_dataset("allenai/sciq", split="test"),
             lambda d: (f"{d['support'].lstrip()}\nQuestion: {d['question']}\nAnswer:",
                        [d["distractor1"], d["distractor2"], d["distractor3"], d["correct_answer"]], 3)),
    "arc_easy": (lambda: load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"),
                 lambda d: (f"Question: {d['question']}\nAnswer:", d["choices"]["text"],
                            d["choices"]["label"].index(d["answerKey"]))),
    "social_iqa": (lambda: load_dataset("baber/social_i_qa", split="validation"),
                   lambda d: (f"Q: {d['context']} {d['question']}\nA:", [d["answerA"], d["answerB"], d["answerC"]],
                              int(d["label"]) - 1)),
    # COPA validation has only 100 examples: add 100 from train (GPT-2 never saw either)
    "copa": (lambda: concatenate_datasets([load_dataset("aps/super_glue", "copa", split=s) for s in ("validation", "train")]),
             _copa),
}


def build() -> None:
    items = []
    for name, (load, fmt) in BENCHMARKS.items():
        ds = load()
        for i in sorted(random.Random(SEED).sample(range(len(ds)), N_EXAMPLES)):
            context, choices, gold = fmt(ds[i])
            items.append({"bench": name, "context": context, "choices": choices, "gold": gold})
    BENCH_FILE.write_text(json.dumps(items, indent=1))
    splits = json.loads(SPLITS_FILE.read_text())
    splits["bench"] = [it["context"] + " " + c for it in items for c in it["choices"]]
    SPLITS_FILE.with_suffix(".tmp").write_text(json.dumps(splits, indent=1))
    SPLITS_FILE.with_suffix(".tmp").replace(SPLITS_FILE)  # atomic: sandbox runs may be reading it
    for m in MODELS:
        add_token_splits(m)
    print(f"{len(items)} examples, {len(splits['bench'])} texts → {BENCH_FILE}")


@torch.no_grad()
def continuation_ll(model, ids: torch.Tensor, n_context: int) -> float:
    """Summed log P(continuation tokens | everything before them)."""
    logp = torch.log_softmax(model(input_ids=ids).logits[0, :-1], -1)
    targets = ids[0, 1:]
    return float(logp[torch.arange(len(targets)), targets][n_context - 1:].sum())


def evaluate(model_name: str = "gpt2") -> dict:
    out_path = run_dir(model_name) / "bench.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    items = json.loads(BENCH_FILE.read_text())
    order, _, assign, matrix, failures = prepare(model_name, "bench")
    counts = [round(f * len(order)) for f in LEVELS]
    tok, model = load_lm(model_name)
    ll = {c: np.zeros((len(LEVELS), sum(len(it["choices"]) for it in items))) for c in assign}
    k, t0 = 0, time.time()
    for j, it in enumerate(items):
        n_context = len(tok(it["context"])["input_ids"])  # lm-eval: continuation = whole[len(context):]
        for choice in it["choices"]:
            ids = tok(it["context"] + " " + choice, return_tensors="pt")["input_ids"]
            n = ids.shape[1]
            base = continuation_ll(model, ids, n_context)
            for cond, a in assign.items():
                ll[cond][0, k] = base
                mats = {lh: matrix(a[lh], k, n) for lh in order}
                for i, count in enumerate(counts[1:], 1):
                    with replaced({lh: mats[lh] for lh in order[:count]}):
                        ll[cond][i, k] = continuation_ll(model, ids, n_context)
            k += 1
        if (j + 1) % 50 == 0:
            print(f"[{j + 1}/{len(items)} {time.time() - t0:5.0f}s]", flush=True)

    result = {"levels": LEVELS, "n_heads": counts, "failed_program_input_pairs": int(failures), "acc": {}, "acc_norm": {}}
    for cond, scores in ll.items():
        for metric in ("acc", "acc_norm"):
            per_bench: dict[str, list] = {}
            k = 0
            for it in items:
                s = scores[:, k:k + len(it["choices"])]
                if metric == "acc_norm":
                    s = s / np.array([len(c) for c in it["choices"]])
                per_bench.setdefault(it["bench"], []).append(s.argmax(1) == it["gold"])
                k += len(it["choices"])
            result[metric][cond] = {b: np.mean(v, 0).tolist() for b, v in per_bench.items()}
    out_path.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":  # required: sandbox children re-import the main module
    if sys.argv[1] == "build":
        build()
        sys.exit()
    r = evaluate(sys.argv[1])
    for b in BENCHMARKS:
        row = {c: [round(100 * x) for x in v[b]] for c, v in r["acc"].items()}
        print(f"{b:11s} acc % per level {LEVELS}:")
        for c, v in row.items():
            print(f"   {c:9s} {v}")
