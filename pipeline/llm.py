"""Synthesis LLM: Claude Haiku 5.5 via the Anthropic API, with a disk cache.

Every request/response is cached by content hash in runs/llm_cache/, so reruns cost nothing.
"""
import hashlib
import json
import os
from functools import cache
from pathlib import Path

import anthropic

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000  # thinking counts toward this; non-streaming stays under SDK HTTP timeouts
EFFORT = "low"  # Haiku 5.5 default is medium; low keeps calls fast and cheap
CACHE_DIR = Path("runs/llm_cache")


def _load_env(path: Path = Path(".env")) -> None:
    """Minimal KEY=VALUE reader; real environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#") and value.strip():
                os.environ.setdefault(key.strip(), value.strip())


@cache
def _client() -> anthropic.Anthropic:
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set: add it to .env")
    return anthropic.Anthropic(max_retries=5)  # SDK retries 429/5xx/connection errors with backoff


def complete(messages: list[dict]) -> dict:
    """{"reply", "stop_reason", "usage": {"input", "output"}, "cached"} for a chat conversation.

    `reply` is the concatenated text blocks; it is "" when the model refused or ran out of tokens
    before writing text (the caller then scores the empty program 0).
    """
    request = {"model": MODEL, "max_tokens": MAX_TOKENS, "output_config": {"effort": EFFORT},
               "messages": messages}
    path = CACHE_DIR / f"{hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}.json"
    if path.exists():
        return {**json.loads(path.read_text()), "cached": True}
    response = _client().messages.create(**request)
    result = {
        # thinking blocks come first (and are empty by default); read text blocks by type
        "reply": "".join(b.text for b in response.content if b.type == "text"),
        "stop_reason": response.stop_reason,
        "usage": {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
    }
    if response.stop_reason == "refusal" and response.stop_details:
        result["refusal_category"] = response.stop_details.category
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**result, "request": request}))
    return {**result, "cached": False}
