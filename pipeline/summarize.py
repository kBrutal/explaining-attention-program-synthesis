"""M4: turn a head's cached attention into the synthesis prompt (paper §2.2 + Appendix A.4).

Run: python -m pipeline.summarize gpt2 0 1   → prints the prompt for GPT-2 L0H1
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from pipeline.extract import MODELS, load_attention

TOP_FRACTION = 0.025  # paper: "top 2.5% of attention weights by magnitude"
ROW_TOP_K = 2  # also keep each query's 2 strongest keys (see kept_edges)
SHOWN_PER_BAND = 3
PROMPT_TOKEN_BUDGET = 4000  # paper: prompts of ~4,000 tokens
CHARS_PER_TOKEN = 3.5  # rough estimate; fine for budgeting examples
BANDS = [("high", 75, 100), ("upper-mid", 50, 75), ("lower-mid", 25, 50), ("low", 0, 25)]
TEMPLATE = Path("prompts/synth.txt")

MODEL_TEXT = {
    "gpt2": {"display": "GPT2-small", "token_kind": "GPT2 BPE", "p": "gpt2",
             "mask_rule": "   - The matrix must be lower-triangular (causal mask): token i can only attend to tokens j <= i",
             "finalize": "`make_row_stochastic()` and `apply_causal_mask()`"},
    "bert-base-uncased": {"display": "BERT-base-uncased", "token_kind": "BERT WordPiece (including [CLS] and [SEP])",
                          "p": "bert",
                          "mask_rule": "   - The model is bidirectional: token i may attend to any token j",
                          "finalize": "`make_row_stochastic()`"},
}

HELPER_DOCS = """\
- `{p}_tokenize(sentence: str) -> list[str]`: the model's token strings for the sentence, exactly as the head sees them.
- `spacy_parse(sentence: str) -> spacy.tokens.Doc`: cached spaCy parse (POS tags, dependencies, entities). `nlp` (en_core_web_sm) is also available.
- `align_{p}_to_spacy(sentence: str) -> tuple[Doc, list[list[int]]]`: (doc, mapping) where mapping[i] lists the spaCy token indices overlapping model token i ([] for special tokens).
- `align_spacy_to_{p}(sentence: str) -> tuple[Doc, list[list[int]]]`: (doc, mapping) where mapping[k] lists the model token indices overlapping spaCy token k.
- `embedding_similarity(tok_a: str, tok_b: str) -> float`: cosine similarity of the model's input embeddings for two token strings.
- `make_row_stochastic(A) -> np.ndarray`: clip negatives and normalize rows to sum to 1 (all-zero rows become uniform).
{causal_doc}- `np` (numpy) and `nltk` are pre-imported."""
CAUSAL_DOC = "- `apply_causal_mask(A) -> np.ndarray`: zero entries with j > i and renormalize rows (all-zero rows become uniform over j <= i).\n"


def kept_edges(A: np.ndarray, causal: bool) -> list[tuple[int, int, float]]:
    """Each row's top-2 keys plus the top 2.5% of all allowed entries, sorted by weight (desc).

    Top-2 per row: the paper's own appendix example shows second-strongest edges
    ('.'[10] -> '!"'[4] (0.022)) that a pure top-2.5% filter would drop.
    """
    n = len(A)
    allowed = np.tril(np.ones((n, n), bool)) if causal else np.ones((n, n), bool)
    masked = np.where(allowed, A, -1.0)
    keep = allowed & (A >= np.quantile(A[allowed], 1 - TOP_FRACTION))
    for k in np.argsort(-masked, axis=1)[:, :ROW_TOP_K].T:
        keep[np.arange(n), k] |= allowed[np.arange(n), k]
    i, j = np.nonzero(keep)
    return sorted(zip(i.tolist(), j.tolist(), A[i, j].tolist()), key=lambda e: -e[2])


def summarize_sentence(sentence: str, tokens: list[str], A: np.ndarray, causal: bool,
                       show_sentence: bool = True) -> str:
    """One example block: kept edges split by rank into quartile bands, a few edges shown per band."""
    edges = kept_edges(A, causal)
    lines = [f'Sentence: "{sentence}"', ""] if show_sentence else []
    # edges are sorted desc, so the first quarter is p75-p100, the last p0-p25
    for (name, lo, hi), band in zip(BANDS, np.array_split(np.arange(len(edges)), 4)):
        band = [edges[k] for k in band]
        if not band:
            continue
        # evenly spaced picks so the band's range (top to bottom) is visible
        shown = [band[k] for k in sorted(set(np.linspace(0, len(band) - 1, min(SHOWN_PER_BAND, len(band)))
                                             .round().astype(int)))]
        lines.append(f"{name} attention (p{lo}-p{hi}, {band[-1][2]:.3f}-{band[0][2]:.3f}):")
        lines += [f"  '{tokens[i]}'[{i}] -> '{tokens[j]}'[{j}] ({w:.3f})" for i, j, w in shown]
        rest = [e for e in band if e not in shown]
        n_self = sum(i == j for i, j, _ in rest)
        if n_self:
            lines.append(f"  (+ {n_self} more self-attention edges)")
        if len(rest) - n_self:
            lines.append(f"  (+ {len(rest) - n_self} more edges)")
    return "\n".join(lines)


def build_prompt(model_name: str, layer: int, head: int, synth=None) -> str:
    """Synthesis prompt for one head; example sentences are a per-head seeded sample of the synth split."""
    sentences = json.loads(Path("data/splits.json").read_text())["synth"]
    synth = synth if synth is not None else load_attention(model_name, "synth")
    text = MODEL_TEXT[model_name]
    causal = MODELS[model_name]["causal"]
    helpers = HELPER_DOCS.format(p=text["p"], causal_doc=CAUSAL_DOC if causal else "")
    fill = dict(layer=layer, head=head, model_display=text["display"], token_kind=text["token_kind"],
                p=text["p"], mask_rule=text["mask_rule"], finalize=text["finalize"], helpers=helpers)
    base_chars = len(TEMPLATE.read_text().format(examples="", **fill))

    order = random.Random(f"{model_name}-L{layer}H{head}").sample(range(len(sentences)), len(sentences))
    blocks, chars = [], base_chars
    for k in order:
        tokens, A = synth[k]
        block = f"Example {len(blocks) + 1}:\n" + summarize_sentence(sentences[k], tokens, A[layer, head], causal)
        if blocks and (chars + len(block)) / CHARS_PER_TOKEN > PROMPT_TOKEN_BUDGET:
            break
        blocks.append(block)
        chars += len(block) + 2
    return TEMPLATE.read_text().format(examples="\n\n".join(blocks), **fill)


if __name__ == "__main__":
    model, layer, head = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    print(build_prompt(model, layer, head))
