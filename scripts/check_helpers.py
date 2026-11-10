"""M3 check: helpers behave, and hand-written programs validate and fit the heads they should."""
import json
from pathlib import Path

import numpy as np

from helpers.attention_helpers import apply_causal_mask, make_row_stochastic, program_namespace
from pipeline.extract import load_attention
from pipeline.score import iou, validate

# --- helper unit checks ---
A = apply_causal_mask(np.ones((4, 4)))
assert np.allclose(A.sum(1), 1) and np.allclose(np.triu(A, 1), 0) and np.isclose(A[3, 0], 0.25)
assert np.allclose(apply_causal_mask(np.zeros((3, 3)))[2], 1 / 3)
assert np.allclose(make_row_stochastic([[0, 0], [-1, 3]]), [[0.5, 0.5], [0, 1]])
assert iou(A, A) == 1 and iou(A, np.zeros_like(A)) == 0 and iou(A, A.T) == iou(A.T, A)

ns = program_namespace("gpt2")
s = 'He said, "Don\'t go!"  Lily  unbelievably smiled.'
doc, g2s = ns["align_gpt2_to_spacy"](s)
toks = ns["gpt2_tokenize"](s)
assert len(g2s) == len(toks)
assert [doc[k].text for k in g2s[toks.index(" Lily")]] == ["Lily"]
assert [doc[k].text for k in g2s[toks.index("Don")]] == ["Do", "n't"]  # 'Don' spans both spaCy tokens
_, b2s = program_namespace("bert-base-uncased")["align_bert_to_spacy"](s)
assert b2s[0] == [] and b2s[-1] == []  # [CLS], [SEP]
sim = ns["embedding_similarity"]
assert sim(" dog", " dog") > 0.99 and sim(" dog", " cat") > sim(" dog", " the")

# --- hand-written programs, run the way synthesized ones will be ---
PROGRAMS = {
    "first_token": """
def predict_attention_map(sentence):
    tokens = gpt2_tokenize(sentence)
    A = np.zeros((len(tokens), len(tokens)))
    A[:, 0] = 1
    return tokens, apply_causal_mask(A)
""",
    "previous_token": """
def predict_attention_map(sentence):
    tokens = gpt2_tokenize(sentence)
    n = len(tokens)
    A = np.eye(n, k=-1)
    A[0, 0] = 1
    return tokens, make_row_stochastic(apply_causal_mask(A))
""",
}
sentences = json.loads(Path("data/splits.json").read_text())["test"]
data = load_attention("gpt2", "test")
for name, src in PROGRAMS.items():
    env = dict(ns)
    exec(src, env)
    preds = [validate(env["predict_attention_map"](sent), toks, causal=True)
             for sent, (toks, _) in zip(sentences, data)]
    scores = np.array([[np.mean([iou(Ak[l, h], P) for (_, Ak), P in zip(data, preds)])
                        for h in range(12)] for l in range(12)])
    l, h = np.unravel_index(scores.argmax(), scores.shape)
    print(f"{name}: best head L{l}H{h} IoU={scores[l, h]:.2f}, heads with IoU>0.5: {(scores > 0.5).sum()}/144")
    if name == "previous_token":
        print(f"  L4H11 IoU={scores[4, 11]:.2f}")
        assert scores[4, 11] > 0.8

print("OK")
