"""Functions shared by extraction (M2) and synthesized programs (M3).

Token strings must be identical on both sides, or IoU compares misaligned matrices.
Programs never import this module: they get `program_namespace(model_name)`, a dict of
pre-imported names (np, nltk, nlp + model-specific helpers like `gpt2_tokenize`).
"""
from functools import cache, lru_cache

import nltk
import numpy as np
import spacy
from transformers import AutoModel, AutoTokenizer

# short names used in helper function names shown to the LLM
PREFIX = {"gpt2": "gpt2", "bert-base-uncased": "bert"}


def token_strings(tokenizer, ids) -> list[str]:
    """Per-token display strings: GPT-2 → ' everywhere' (leading space kept), BERT → '[CLS]', '##bel'."""
    return [tokenizer.decode([i]) for i in ids]


@cache
def _tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


@cache
def _nlp():
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=4096)
def spacy_parse(sentence: str):
    """spaCy Doc (POS, dependencies, NER). Cached: programs re-parse the same sentences many times."""
    return _nlp()(sentence)


@lru_cache(maxsize=4096)
def _encode(model_name: str, sentence: str):
    enc = _tokenizer(model_name)(sentence, return_offsets_mapping=True)
    return token_strings(_tokenizer(model_name), enc["input_ids"]), enc["offset_mapping"]


def tokenize(model_name: str, sentence: str) -> list[str]:
    return list(_encode(model_name, sentence)[0])


def align_to_spacy(model_name: str, sentence: str):
    """(doc, mapping): mapping[i] = spaCy token indices overlapping model token i ([] for special tokens)."""
    doc = spacy_parse(sentence)
    spans = [(t.idx, t.idx + len(t.text)) for t in doc]
    mapping = []
    for start, end in _encode(model_name, sentence)[1]:
        mapping.append([k for k, (s, e) in enumerate(spans)
                        if start < end and s < end and start < e and not doc[k].is_space])
    return doc, mapping


def align_from_spacy(model_name: str, sentence: str):
    """(doc, mapping): mapping[k] = model token indices overlapping spaCy token k."""
    doc, to_spacy = align_to_spacy(model_name, sentence)
    mapping = [[] for _ in doc]
    for i, ks in enumerate(to_spacy):
        for k in ks:
            mapping[k].append(i)
    return doc, mapping


def make_row_stochastic(A) -> np.ndarray:
    """Clip negatives, normalize rows to sum to 1; all-zero rows become uniform."""
    A = np.clip(np.nan_to_num(np.asarray(A, dtype=np.float64)), 0, None)
    sums = A.sum(-1, keepdims=True)
    return np.where(sums > 0, A / np.where(sums > 0, sums, 1), 1.0 / A.shape[-1])


def apply_causal_mask(A) -> np.ndarray:
    """Zero entries with j > i, then renormalize rows; all-zero rows become uniform over j <= i."""
    A = np.tril(np.clip(np.nan_to_num(np.asarray(A, dtype=np.float64)), 0, None))
    sums = A.sum(-1, keepdims=True)
    uniform = np.tril(np.ones_like(A)) / np.arange(1, A.shape[-1] + 1)[:, None]
    return np.where(sums > 0, A / np.where(sums > 0, sums, 1), uniform)


@cache
def _embeddings(model_name: str) -> np.ndarray:
    W = AutoModel.from_pretrained(model_name).get_input_embeddings().weight.detach().numpy()
    return W / np.linalg.norm(W, axis=1, keepdims=True)


@lru_cache(maxsize=65536)
def embedding_similarity(model_name: str, tok_a: str, tok_b: str) -> float:
    """Cosine similarity of the target model's input embeddings (mean over sub-tokens)."""
    tok, E = _tokenizer(model_name), _embeddings(model_name)

    def vec(s):
        ids = tok.convert_tokens_to_ids([s]) if s in tok.all_special_tokens else \
            tok(s, add_special_tokens=False)["input_ids"]
        return E[ids].mean(0) if ids else np.zeros(E.shape[1])

    a, b = vec(tok_a), vec(tok_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0


def program_namespace(model_name: str) -> dict:
    """Globals a synthesized program runs with. Helper names are model-specific (gpt2_tokenize, ...)."""
    p = PREFIX[model_name]
    ns = {
        "np": np, "nltk": nltk, "nlp": _nlp(), "spacy_parse": spacy_parse,
        f"{p}_tokenize": lambda sentence: tokenize(model_name, sentence),
        f"align_{p}_to_spacy": lambda sentence: align_to_spacy(model_name, sentence),
        f"align_spacy_to_{p}": lambda sentence: align_from_spacy(model_name, sentence),
        "embedding_similarity": lambda a, b: embedding_similarity(model_name, a, b),
        "make_row_stochastic": make_row_stochastic,
    }
    if model_name == "gpt2":  # BERT is bidirectional: no causal mask helper
        ns["apply_causal_mask"] = apply_causal_mask
    return ns
