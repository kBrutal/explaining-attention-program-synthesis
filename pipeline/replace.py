"""M11: replace attention heads with program outputs inside the model's forward pass.

A custom attention function (registered with transformers' AttentionInterface) runs normal eager
attention, then for each replaced head overwrites its attention matrix with the program's
prediction (or with zeros: zero-ablation / "pruning", which zeroes that head's output) and recomputes
attn @ V. The rest of the layer (c_proj, residual, MLP) is untouched.

Run the self-checks: python -m pipeline.replace
"""
import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward

# layer → {head: [n, n] tensor, or None for zero-ablation}. Set per input by `replaced(...)`.
REPLACED: dict[int, dict[int, torch.Tensor | None]] = {}


def patched_attention(module, query, key, value, attention_mask, **kwargs):
    out, weights = eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
    heads = REPLACED.get(module.layer_idx)
    if not heads:
        return out, weights  # untouched: bit-identical to eager
    weights = weights.clone()
    for h, P in heads.items():
        weights[:, h] = 0 if P is None else P
    return torch.matmul(weights, value).transpose(1, 2), weights


AttentionInterface.register("replace", patched_attention)
# a custom attention name otherwise gets no causal mask (None): reuse eager's additive mask
AttentionMaskInterface.register("replace", ALL_MASK_ATTENTION_FUNCTIONS["eager"])


class replaced:
    """Context manager: `with replaced({(layer, head): matrix or None}): model(...)`."""

    def __init__(self, heads: dict[tuple[int, int], torch.Tensor | None]):
        self.heads = heads

    def __enter__(self):
        for (l, h), P in self.heads.items():
            REPLACED.setdefault(l, {})[h] = P

    def __exit__(self, *_):
        REPLACED.clear()


def load_lm(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="replace").eval()
    return tok, model


@torch.no_grad()
def nll(model, ids: torch.Tensor) -> tuple[float, int]:
    """(summed next-token negative log-likelihood, number of predicted tokens) for one [1, n] input."""
    logits = model(input_ids=ids).logits[0, :-1]
    return float(torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")), ids.shape[1] - 1


@torch.no_grad()
def _checks():
    tok, model = load_lm("gpt2")
    ref = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
    ids = tok("Once upon a time, there was a little girl named Lily. She loved to play outside.",
              return_tensors="pt")["input_ids"]
    base = ref(input_ids=ids).logits
    assert torch.equal(model(input_ids=ids).logits, base), "no replacement must be bit-identical"

    true = ref(input_ids=ids, output_attentions=True).attentions  # [1, H, n, n] per layer
    with replaced({(5, 3): true[5][0, 3], (0, 0): true[0][0, 0]}):
        assert torch.allclose(model(input_ids=ids).logits, base, atol=1e-5), "own attention must not change logits"
    with replaced({(5, 3): None}):
        assert not torch.allclose(model(input_ids=ids).logits, base, atol=1e-3), "pruning must change logits"
    assert not REPLACED, "context manager must clean up"
    print("OK: unpatched bit-identical, own-attention replacement unchanged, pruning changes logits")


if __name__ == "__main__":
    _checks()
