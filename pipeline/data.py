"""M1: TinyStories → fixed, seeded, disjoint sentence splits.

synth/val sentences come from the TinyStories train split, test sentences and
perplexity stories from the validation split. Writes data/splits.json.
"""
import json
import random
from pathlib import Path

import spacy
from datasets import load_dataset

SEED = 0
SIZES = {"synth": 100, "val": 50, "test": 200}  # sentences
N_PPL_STORIES = 200
N_STORIES, STORY_MAX_WORDS = 50, 300  # whole-story IoU split: fits BERT's 512 tokens untruncated
MIN_TOKENS = 5  # spaCy tokens; shorter sentences give near-trivial attention maps
STORY_POOL = 2000  # stories sampled per source split before sentence splitting
OUT = Path("data/splits.json")


def is_mojibake(text: str) -> bool:
    """Some TinyStories texts are UTF-8 mis-decoded as cp1252 (e.g. 'â€œDonâ€™t')."""
    return "â€" in text or "Ã" in text


def sample_stories(split: str, n: int, rng: random.Random) -> list[str]:
    """n random stories, skipping mojibake ones (dropped, not repaired: no text preprocessing)."""
    ds = load_dataset("roneneldan/TinyStories", split=split)
    out = []
    for i in rng.sample(range(len(ds)), len(ds)):
        if not is_mojibake(text := ds[i]["text"]):
            out.append(text)
            if len(out) == n:
                return out
    raise ValueError(f"only {len(out)} clean stories in {split}")


def to_sentences(stories: list[str], nlp) -> list[str]:
    """Unique sentences (first-seen order) with at least MIN_TOKENS tokens."""
    seen = {}
    # spaCy's senter sometimes merges sentences across blank lines; paragraphs are hard boundaries
    paragraphs = [p for s in stories for p in s.split("\n\n") if p.strip()]
    for doc in nlp.pipe(paragraphs):
        for sent in doc.sents:
            text = sent.text.strip()
            if len(sent) >= MIN_TOKENS and text not in seen:
                seen[text] = None
    return list(seen)


def build_splits() -> dict:
    rng = random.Random(SEED)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp.enable_pipe("senter")

    train_sents = to_sentences(sample_stories("train", STORY_POOL, rng), nlp)
    val_stories = sample_stories("validation", STORY_POOL + N_PPL_STORIES, rng)
    ppl_stories, test_pool = val_stories[:N_PPL_STORIES], val_stories[N_PPL_STORIES:]
    # test must not repeat any train-side sentence (TinyStories reuses stock phrases)
    test_sents = [s for s in to_sentences(test_pool, nlp) if s not in set(train_sents)]

    rng.shuffle(train_sents)
    rng.shuffle(test_sents)
    n_s, n_v = SIZES["synth"], SIZES["val"]
    return {
        "seed": SEED,
        "synth": train_sents[:n_s],
        "val": train_sents[n_s:n_s + n_v],
        "test": test_sents[:SIZES["test"]],
        "ppl_stories": ppl_stories,
        "stories": (stories := [s for s in ppl_stories if len(s.split()) <= STORY_MAX_WORDS][:N_STORIES]),
        "ppl_eval": [s for s in ppl_stories if s not in stories],  # M12 perplexity; never used for selection
    }


if __name__ == "__main__":
    splits = build_splits()
    sets = [set(splits[k]) for k in SIZES]
    assert all(len(splits[k]) == n for k, n in SIZES.items()), {k: len(splits[k]) for k in SIZES}
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:]), "splits overlap"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(splits, indent=1))
    print({k: len(v) for k, v in splits.items() if k != "seed"}, "→", OUT)
