"""Five-bin strength math: normalize, expectation, entropy.

Per insight.md, an LLM emits level_probs = [P(level0)..P(level4)];
the engine derives strength = sum_k p(k) * k and entropy H = -sum p log2 p.
The graph stores only `strength`; distributions live in the session log
so sampling-diversity experiments stay possible later.
"""

from __future__ import annotations

import math

BINS = 5


def normalize5(probs: list[float]) -> tuple[list[float], str | None]:
    """Normalize a 5-vector to sum to 1. Returns (probs, note) where note
    describes the normalization applied (None if it was already clean).

    A zero-sum vector maps to [1,0,0,0,0] (mass at level 0 = absent).
    """
    total = sum(probs)
    if total <= 1e-9:
        return [1.0, 0.0, 0.0, 0.0, 0.0], "level_probs summed to ~0; set to [1,0,0,0,0]"
    if abs(total - 1.0) > 1e-6:
        note = f"normalized level_probs (sum was {total:.3f})"
        return [x / total for x in probs], note
    return list(probs), None


def strength_of(probs: list[float]) -> float:
    """strength = sum_k p(k) * k, rounded to 3 decimals."""
    return round(sum(k * p for k, p in enumerate(probs)), 3)


def entropy_of(probs: list[float]) -> float:
    """Shannon entropy in bits over the normalized 5-bin distribution."""
    return round(-sum(p * math.log2(p) for p in probs if p > 0.0), 3)


def clamp(value: float, lo: float, hi: float) -> tuple[float, str | None]:
    """Clamp value into [lo, hi]; return (value, note) with note when clamped."""
    if value < lo:
        return lo, f"clamped {value} to [{lo},{hi}]"
    if value > hi:
        return hi, f"clamped {value} to [{lo},{hi}]"
    return value, None
