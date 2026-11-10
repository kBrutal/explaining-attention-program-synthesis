"""M6 check without network: fake LLM replies, real sandbox/scoring/selection/record writing."""
import shutil

from pipeline import llm, synthesize
from pipeline.extract import run_dir

MODEL, LAYER, HEAD = "gpt2", 4, 11  # previous-token head

FIRST = """Here is my attempt:
```python
def predict_attention_map(sentence):
    tokens = gpt2_tokenize(sentence)
    A = np.zeros((len(tokens), len(tokens)))
    A[:, 0] = 1
    return tokens, apply_causal_mask(A)
```"""
SECOND = """```python
def predict_attention_map(sentence):
    tokens = gpt2_tokenize(sentence)
    A = np.eye(len(tokens), k=-1)
    A[0, 0] = 1
    return tokens, apply_causal_mask(A)
```"""


def main():
    sent = []

    def fake_complete(messages):
        sent.append(messages)
        return {"reply": FIRST if len(sent) == 1 else SECOND, "stop_reason": "end_turn",
                "usage": {"input": 0, "output": 0}, "cached": False}

    llm.complete = fake_complete
    record_dir = run_dir(MODEL) / "programs"
    # move a real M7 record aside (synthesize_head would skip the head) and put it back afterwards
    saved = {sfx: (record_dir / f"L{LAYER}H{HEAD}{sfx}").read_text()
             for sfx in (".json", ".py") if (record_dir / f"L{LAYER}H{HEAD}{sfx}").exists()}
    for sfx in saved:
        (record_dir / f"L{LAYER}H{HEAD}{sfx}").unlink()
    try:
        r = synthesize.synthesize_head(MODEL, LAYER, HEAD)
        c0, c1 = r["candidates"]
        assert len(sent) == 2 and len(sent[1]) == 3  # prompt, program, feedback
        feedback = sent[1][2]["content"]
        assert "BEST example" in feedback and "WORST example" in feedback and "YOUR PREDICTED" in feedback
        assert c0["val_iou"] < 0.3 < 0.8 < c1["val_iou"] and r["chosen"] == 1, r
        assert (record_dir / f"L{LAYER}H{HEAD}.py").read_text().strip() == c1["src"]
        print(f"first-token {c0['val_iou']:.3f} → refined previous-token {c1['val_iou']:.3f}, chosen={r['chosen']}")
        print("--- feedback prompt (start) ---\n" + feedback[:900])
    finally:  # remove the fake record, restore the real one if there was one
        for suffix in (".json", ".py"):
            (record_dir / f"L{LAYER}H{HEAD}{suffix}").unlink(missing_ok=True)
        for sfx, text in saved.items():
            (record_dir / f"L{LAYER}H{HEAD}{sfx}").write_text(text)
        if record_dir.exists() and not any(record_dir.iterdir()):
            shutil.rmtree(record_dir)
    print("OK")


if __name__ == "__main__":  # required: sandbox children re-import the main module
    main()
