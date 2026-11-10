"""M5: run a synthesized program on a split in a subprocess, validate, cache, and score.

Isolation guards against accidents (hangs, crashes, stray network/file use), not malicious code:
separate process, per-sentence alarm, overall kill timeout, sockets disabled, temp cwd.
"""
import os

# Must precede any transformers import: HF reads these at import time, and spawned sandbox children
# import this module first. Models are cached locally, and children have sockets disabled anyway.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from pipeline.extract import MODELS, load_attention, run_dir  # noqa: E402
from pipeline.score import validate  # noqa: E402

PER_SENTENCE_TIMEOUT = 5  # seconds
MAX_TIMEOUTS_IN_A_ROW = 3  # then fail the remaining sentences: the program is almost surely stuck
STARTUP_TIMEOUT = 120  # seconds: imports + spaCy load in the child
SPLITS_FILE = Path("data/splits.json")


def extract_code(reply: str) -> str | None:
    """The single ```python block from an LLM reply (last one if several), else None."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    return blocks[-1].strip() if blocks else None


def program_id(model_name: str, src: str) -> str:
    return hashlib.sha256(f"{model_name}\n{src}".encode()).hexdigest()[:16]


def _child(conn, model_name: str, src: str, sentences: list[str]) -> None:
    """Runs in the subprocess: exec the program, call it on each sentence, send back raw outputs."""
    import signal
    import socket
    import tempfile

    def no_network(*args, **kwargs):
        raise PermissionError("network access is disabled for programs")

    socket.socket = no_network
    socket.create_connection = no_network
    os.chdir(tempfile.mkdtemp())

    def alarm(signum, frame):
        raise TimeoutError(f"exceeded {PER_SENTENCE_TIMEOUT}s")

    signal.signal(signal.SIGALRM, alarm)
    from helpers.attention_helpers import program_namespace, spacy_parse, tokenize

    env = program_namespace(model_name)
    tokenize(model_name, "warm up")  # load tokenizer + spaCy outside the per-sentence timer
    spacy_parse("warm up")
    try:
        exec(compile(src, "<program>", "exec"), env)
        fn = env["predict_attention_map"]
    except BaseException as e:  # syntax error, missing function, ...
        conn.send(("program_error", f"{type(e).__name__}: {e}"))
        return
    results, timeouts_in_a_row = [], 0
    for sentence in sentences:
        if timeouts_in_a_row >= MAX_TIMEOUTS_IN_A_ROW:
            results.append(("error", f"skipped after {MAX_TIMEOUTS_IN_A_ROW} timeouts in a row"))
            continue
        signal.alarm(PER_SENTENCE_TIMEOUT)
        try:
            tokens, A = fn(sentence)
            results.append(("ok", ([str(t) for t in tokens], np.asarray(A, dtype=np.float64))))
            timeouts_in_a_row = 0
        except BaseException as e:
            results.append(("error", f"{type(e).__name__}: {e}"))
            timeouts_in_a_row = timeouts_in_a_row + 1 if isinstance(e, TimeoutError) else 0
        finally:
            signal.alarm(0)
    conn.send(("done", results))


def run_program(model_name: str, src: str, split: str) -> tuple[list[np.ndarray | None], list[str | None]]:
    """(matrices, errors) per sentence of `split`; failed sentences have matrix None and an error message.

    Validated outputs are cached at runs/<model>/outputs/<program_id>/<split>.npz.
    """
    cache_dir = run_dir(model_name) / "outputs" / program_id(model_name, src)
    cache = cache_dir / f"{split}.npz"
    sentences = json.loads(SPLITS_FILE.read_text())[split]
    if cache.exists():
        errors = json.loads((cache_dir / f"{split}_errors.json").read_text())
        with np.load(cache) as z:
            return [z[str(i)].astype(np.float64) if errors[i] is None else None
                    for i in range(len(sentences))], errors

    expected = json.loads((run_dir(model_name) / "tokens.json").read_text())[split]
    raw = _run_in_subprocess(model_name, src, sentences)
    causal = MODELS[model_name]["causal"]
    mats, errors = [], []
    for k, (status, payload) in enumerate(raw):
        try:
            if status != "ok":
                raise ValueError(payload)
            # round like the float16 cache so fresh and cached runs give identical scores and prompts
            mats.append(validate(tuple(payload), expected[k], causal).astype(np.float16).astype(np.float64))
            errors.append(None)
        except ValueError as e:
            mats.append(None)
            errors.append(str(e))

    # write-then-rename: parallel heads can produce identical source → same cache dir.
    # errors.json lands before the .npz, so a visible .npz always has its errors file.
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = f".{os.getpid()}.{id(mats)}.tmp"
    (cache_dir / f"program.py{tmp}").write_text(src)
    os.replace(cache_dir / f"program.py{tmp}", cache_dir / "program.py")
    (cache_dir / f"{split}_errors.json{tmp}").write_text(json.dumps(errors))
    os.replace(cache_dir / f"{split}_errors.json{tmp}", cache_dir / f"{split}_errors.json")
    with open(cache_dir / f"{split}.npz{tmp}", "wb") as f:
        np.savez_compressed(f, **{str(i): m.astype(np.float16) for i, m in enumerate(mats) if m is not None})
    os.replace(cache_dir / f"{split}.npz{tmp}", cache)
    return mats, errors


def _run_in_subprocess(model_name: str, src: str, sentences: list[str]) -> list[tuple[str, object]]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, model_name, src, sentences), daemon=True)
    proc.start()
    child.close()
    timeout = STARTUP_TIMEOUT + PER_SENTENCE_TIMEOUT * len(sentences)
    try:
        status, payload = parent.recv() if parent.poll(timeout) else ("program_error", "overall timeout")
    except EOFError:  # child died without sending (segfault, os._exit, ...)
        status, payload = "program_error", f"process died (exit code {proc.exitcode})"
    finally:
        proc.kill()
        proc.join()
    if status == "program_error":
        return [("error", payload)] * len(sentences)
    return payload


def head_ious(mats: list[np.ndarray | None], attn: list[tuple[list[str], np.ndarray]]) -> np.ndarray:
    """Mean IoU over sentences against every head at once → [layers, heads]. Failed sentences count 0."""
    total = np.zeros(attn[0][1].shape[:2])
    for P, (_, A) in zip(mats, attn):
        if P is not None:  # A: [L, H, n, n], P: [n, n]
            total += np.minimum(A, P).sum((-2, -1)) / np.maximum(A, P).sum((-2, -1))
    return total / len(attn)


def score_program(model_name: str, src: str, split: str, attn=None) -> tuple[np.ndarray, list[str | None]]:
    """([layers, heads] mean IoU of the program vs every head on `split`, per-sentence errors)."""
    mats, errors = run_program(model_name, src, split)
    return head_ious(mats, attn if attn is not None else load_attention(model_name, split)), errors
