"""M2: run each target model on data/splits.json and cache every head's attention.

Writes runs/<model>/attn.npz  (key "<split>_<i>" → float16 [layers, heads, n, n])
   and runs/<model>/tokens.json ({split: [[token strings], ...]}).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from helpers.attention_helpers import token_strings

MODELS = {"gpt2": {"causal": True}, "bert-base-uncased": {"causal": False}}
SPLITS = ("synth", "val", "test", "stories")
TOKEN_ONLY_SPLITS = ("ppl_eval", "bench")  # programs run on these (M12, M13); no attention stored
SPLITS_FILE = Path("data/splits.json")


def run_dir(model_name: str) -> Path:
    return Path("runs") / model_name


def load_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    # eager attention: SDPA doesn't return attention probabilities
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager").eval()
    return tok, model


@torch.no_grad()
def attention(tok, model, sentence: str) -> tuple[list[str], np.ndarray]:
    """(token strings, float32 [layers, heads, n, n]) for one sentence, exactly as the model sees it."""
    ids = tok(sentence, return_tensors="pt")["input_ids"]
    attns = model(input_ids=ids, output_attentions=True).attentions
    return token_strings(tok, ids[0].tolist()), torch.cat(attns).numpy()


def extract(model_name: str) -> None:
    splits = json.loads(SPLITS_FILE.read_text())
    tok, model = load_model(model_name)
    causal = MODELS[model_name]["causal"]
    arrays, tokens = {}, {}
    for split in SPLITS:
        tokens[split] = []
        for i, sentence in enumerate(splits[split]):
            toks, A = attention(tok, model, sentence)
            n = len(toks)
            assert A.shape[-2:] == (n, n), (split, i, A.shape, n)
            assert np.allclose(A.sum(-1), 1, atol=1e-4), (split, i)
            if causal:
                assert np.allclose(np.triu(A, 1), 0), (split, i)
            arrays[f"{split}_{i}"] = A.astype(np.float16)
            tokens[split].append(toks)
    out = run_dir(model_name)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "attn.npz", **arrays)
    (out / "tokens.json").write_text(json.dumps(tokens))
    add_token_splits(model_name)
    shape = next(iter(arrays.values())).shape[:2]
    print(f"{model_name}: {len(arrays)} sentences, {shape[0]}x{shape[1]} heads → {out}")


def add_token_splits(model_name: str) -> None:
    """Token strings (no attention) for TOKEN_ONLY_SPLITS present in splits.json, merged into tokens.json."""
    splits, tok = json.loads(SPLITS_FILE.read_text()), AutoTokenizer.from_pretrained(model_name)
    path = run_dir(model_name) / "tokens.json"
    tokens = json.loads(path.read_text())
    for split in TOKEN_ONLY_SPLITS:
        if split in splits:
            tokens[split] = [token_strings(tok, tok(s)["input_ids"]) for s in splits[split]]
    path.with_suffix(".tmp").write_text(json.dumps(tokens))
    path.with_suffix(".tmp").replace(path)  # atomic: sandbox runs read this file concurrently


def load_attention(model_name: str, split: str) -> list[tuple[list[str], np.ndarray]]:
    """[(tokens, float32 [layers, heads, n, n]), ...] for every sentence in a split."""
    d = run_dir(model_name)
    tokens = json.loads((d / "tokens.json").read_text())[split]
    with np.load(d / "attn.npz") as z:
        return [(t, z[f"{split}_{i}"].astype(np.float32)) for i, t in enumerate(tokens)]


if __name__ == "__main__":
    for name in sys.argv[1:] or MODELS:
        extract(name)
