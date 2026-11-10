"""M0 check: models load with eager attention and expose per-head attention probabilities."""
import nltk
import numpy as np
import spacy
import torch
from transformers import AutoModel, AutoTokenizer

MODELS = {"gpt2": True, "bert-base-uncased": False}  # name -> is_causal
SENTENCE = "Lily saw a big dog in the park. She smiled."

spacy.load("en_core_web_sm")
nltk.word_tokenize  # nltk importable; corpora fetched later only if a helper needs them

for name, causal in MODELS.items():
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, attn_implementation="eager").eval()
    inputs = tok(SENTENCE, return_tensors="pt")
    with torch.no_grad():
        attns = model(**inputs, output_attentions=True).attentions
    n = inputs["input_ids"].shape[1]
    heads = sum(a.shape[1] for a in attns)
    assert heads == 144, (name, heads)
    for a in attns:
        a = a[0].numpy()
        assert a.shape[1:] == (n, n)
        assert np.allclose(a.sum(-1), 1, atol=1e-4)
        if causal:
            assert np.allclose(np.triu(a, 1), 0)
    print(f"{name}: {len(attns)} layers x {attns[0].shape[1]} heads = {heads}, n={n}, "
          f"tokens={tok.convert_ids_to_tokens(inputs['input_ids'][0])}")

print("MPS available:", torch.backends.mps.is_available())
print("OK")
