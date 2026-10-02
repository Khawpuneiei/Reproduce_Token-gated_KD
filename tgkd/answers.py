"""Answer extraction, answer-token confidence, and calibration metrics."""

from __future__ import annotations

import math
import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> tuple[str | None, tuple[int, int] | None]:
    """Return (normalised number, char span). Priority: first number after the
    first '####'; else first number inside the last '\boxed{...}' (the teacher's
    native style); else the last number in the text."""
    idx = text.find("####")
    if idx >= 0:
        m = _NUM.search(text, idx + 4)
        if m:
            return _norm(m.group()), m.span()
    idx = text.rfind(r"\boxed{")
    if idx >= 0:
        m = _NUM.search(text, idx + 7)
        close = text.find("}", idx)
        if m and (close < 0 or m.start() < close):
            return _norm(m.group()), m.span()
    matches = list(_NUM.finditer(text))
    if not matches:
        return None, None
    m = matches[-1]
    return _norm(m.group()), m.span()


def _norm(s: str) -> str:
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return s
    return str(int(v)) if v == int(v) else str(v)


def has_final_format(text: str) -> bool:
    return "####" in text or r"\boxed{" in text


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(_norm(gold))) < 1e-6
    except ValueError:
        return pred == gold


def span_token_indices(pieces: list[str], span: tuple[int, int]) -> list[int]:
    """Indices of tokens whose decoded characters overlap [start, end)."""
    out, pos = [], 0
    for i, piece in enumerate(pieces):
        nxt = pos + len(piece)
        if nxt > span[0] and pos < span[1]:
            out.append(i)
        pos = nxt
    return out


def answer_confidence(token_logprobs: list[float], indices: list[int]) -> float:
    """Joint probability of the answer tokens under the model's own greedy decode."""
    if not indices:
        return 0.0
    return math.exp(sum(token_logprobs[i] for i in indices))


def expected_calibration_error(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    n = len(conf)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        avg = sum(conf[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - avg)
    return total


def brier(conf: list[float], correct: list[bool]) -> float:
    return sum((c - float(y)) ** 2 for c, y in zip(conf, correct)) / max(len(conf), 1)
