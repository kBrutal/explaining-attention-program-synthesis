"""M6: synthesize the intended program for one head (paper §2.2).

  π  = LLM(prompt)                      scored on synth (feedback) and val (selection)
  π' = LLM(prompt, π, best/worst-example feedback)
  intended = argmax over {π, π'} of val IoU

Run: python -m pipeline.synthesize gpt2 4 11
Writes runs/<model>/programs/L<l>H<h>.json (both candidates + choice) and L<l>H<h>.py (chosen source).
"""
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pipeline import llm
from pipeline.extract import MODELS, load_attention, run_dir
from pipeline.sandbox import extract_code, head_ious, run_program
from pipeline.summarize import build_prompt, summarize_sentence

N_BEST, N_WORST = 2, 2  # feedback examples
REFINE_TEMPLATE = Path("prompts/refine.txt")
SANDBOX_SLOTS = threading.Semaphore(4)  # ~0.5 GB RAM per sandbox child; 16 GB laptop
PAPER_INTENDED_IOU = {"gpt2": "mean 0.46 (Table A.6)", "bert-base-uncased": "median ~0.22 (Fig. 3)"}
DEFAULT_WORKERS = 16  # heads in flight at once; far below Haiku 5.5 rate limits


def sandboxed(model_name: str, src: str, split: str):
    """run_program, limited to SANDBOX_SLOTS concurrent subprocesses (each loads torch + spaCy)."""
    with SANDBOX_SLOTS:
        return run_program(model_name, src, split)


def evaluate(model_name: str, src: str, layer: int, head: int, attn: dict) -> dict:
    """Run a candidate on synth + val; per-sentence synth IoUs feed the refinement prompt."""
    out: dict = {"src": src}
    for split in ("synth", "val"):
        mats, errors = sandboxed(model_name, src, split)
        out[f"{split}_iou"] = float(head_ious(mats, attn[split])[layer, head])
        out[f"{split}_errors"] = errors
        if split == "synth":
            out["synth_mats"] = mats
    return out


def feedback_prompt(model_name: str, layer: int, head: int, cand: dict, attn: dict) -> str:
    sentences = json.loads(Path("data/splits.json").read_text())["synth"]
    causal = MODELS[model_name]["causal"]
    per_sentence = [  # failed sentences score 0, like in head_ious
        0.0 if P is None else float(np.minimum(A[layer, head], P).sum() / np.maximum(A[layer, head], P).sum())
        for P, (_, A) in zip(cand["synth_mats"], attn["synth"])
    ]
    ok = [k for k, P in enumerate(cand["synth_mats"]) if P is not None]
    ranked = sorted(ok, key=lambda k: -per_sentence[k])
    picks = ranked[:N_BEST] + [k for k in ranked[-N_WORST:] if k not in ranked[:N_BEST]]

    blocks = []
    for k in picks:
        tokens, A = attn["synth"][k]
        tag = "BEST" if k in ranked[:N_BEST] else "WORST"
        blocks.append(
            f"### {tag} example (IoU {per_sentence[k]:.3f})\n"
            f"REAL attention:\n{summarize_sentence(sentences[k], tokens, A[layer, head], causal)}\n\n"
            f"YOUR PREDICTED attention:\n"
            f"{summarize_sentence(sentences[k], tokens, cand['synth_mats'][k], causal, show_sentence=False)}")

    distinct = sorted({e for e in cand["synth_errors"] if e})
    n_failed = sum(e is not None for e in cand["synth_errors"])
    errors = (f"\nIt FAILED on {n_failed} of {len(sentences)} sentences (failures score 0). Errors:\n"
              + "\n".join(f"- {e[:300]}" for e in distinct[:3]) + "\n") if n_failed else ""
    if not blocks:
        blocks = ["(no successful predictions to show; fix the errors above first)"]
    return REFINE_TEMPLATE.read_text().format(score=cand["synth_iou"], n_sentences=len(sentences),
                                              errors=errors, examples="\n\n".join(blocks))


def synthesize_head(model_name: str, layer: int, head: int, attn: dict | None = None) -> dict:
    """Generate, refine once, select on val. Skips heads that already have a record."""
    out_dir = run_dir(model_name) / "programs"
    record_path = out_dir / f"L{layer}H{head}.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    attn = attn or {s: load_attention(model_name, s) for s in ("synth", "val", "test")}

    messages = [{"role": "user", "content": build_prompt(model_name, layer, head, attn["synth"])}]
    first = llm.complete(messages)
    cand0 = evaluate(model_name, extract_code(first["reply"]) or "", layer, head, attn)

    messages += [{"role": "assistant", "content": f"```python\n{cand0['src']}\n```"},
                 {"role": "user", "content": feedback_prompt(model_name, layer, head, cand0, attn)}]
    second = llm.complete(messages)
    cand1 = evaluate(model_name, extract_code(second["reply"]) or "", layer, head, attn)

    candidates = [cand0, cand1]
    chosen = int(np.argmax([c["val_iou"] for c in candidates]))  # ties → initial program
    test_mats, _ = sandboxed(model_name, candidates[chosen]["src"], "test")  # reported number (M7)
    record = {
        "model": model_name, "layer": layer, "head": head, "llm": llm.MODEL,
        "effort": llm.EFFORT, "chosen": chosen,
        "test_iou": float(head_ious(test_mats, attn["test"])[layer, head]),
        "candidates": [{"src": c["src"], "synth_iou": c["synth_iou"], "val_iou": c["val_iou"],
                        "n_val_errors": sum(e is not None for e in c["val_errors"]),
                        "val_error_sample": next((e for e in c["val_errors"] if e), None)}
                       for c in candidates],
        "usage": [first["usage"], second["usage"]],
        "stop_reasons": [first["stop_reason"], second["stop_reason"]],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=1))
    (out_dir / f"L{layer}H{head}.py").write_text(candidates[chosen]["src"] + "\n")
    return record


def synthesize_all(model_name: str, workers: int = DEFAULT_WORKERS) -> None:
    """M7: every head of a model, `workers` at a time. Resumable; a failed head is logged and skipped."""
    attn = {s: load_attention(model_name, s) for s in ("synth", "val", "test")}
    n_layers, n_heads = attn["synth"][0][1].shape[:2]
    heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    t0, done, failed = time.time(), [], []
    print(f"{model_name}: {len(heads)} heads, {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(synthesize_head, model_name, l, h, attn): (l, h) for l, h in heads}
        for fut in as_completed(futures):
            l, h = futures[fut]
            try:
                r = fut.result()
                done.append(r)
                print(f"[{len(done) + len(failed)}/{len(heads)} {time.time() - t0:5.0f}s] L{l}H{h} "
                      f"test IoU {r['test_iou']:.3f} (val {[round(c['val_iou'], 3) for c in r['candidates']]}, "
                      f"chosen {r['chosen']})", flush=True)
            except Exception:
                failed.append((l, h))
                print(f"[FAILED] L{l}H{h}\n{traceback.format_exc()}", flush=True)
    usage = [u for r in done for u in r["usage"]]
    print(f"\ndone {len(done)}, failed {len(failed)} {failed}")
    if done:
        print(f"mean intended test IoU {np.mean([r['test_iou'] for r in done]):.3f} "
              f"(paper: {PAPER_INTENDED_IOU[model_name]}) | "
              f"refined chosen for {np.mean([r['chosen'] for r in done]):.0%} of heads")
        print(f"tokens: input {sum(u['input'] for u in usage):,}, output {sum(u['output'] for u in usage):,}")


if __name__ == "__main__":  # required: sandbox children re-import the main module
    if sys.argv[2] == "all":  # python -m pipeline.synthesize gpt2 all [workers]
        synthesize_all(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_WORKERS)
        sys.exit()
    m, l, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    r = synthesize_head(m, l, h)
    for i, c in enumerate(r["candidates"]):
        print(f"candidate {i}: synth IoU {c['synth_iou']:.3f}, val IoU {c['val_iou']:.3f}, "
              f"val errors {c['n_val_errors']}" + ("  ← chosen" if i == r["chosen"] else ""))
    print(f"test IoU (chosen): {r['test_iou']:.3f} | usage: {r['usage']}")
