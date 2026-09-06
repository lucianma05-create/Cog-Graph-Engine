"""Numeric guards. clamp() clips appraisal/emotion scalars to their ranges."""

from __future__ import annotations


def clamp(v: float, lo: float, hi: float) -> tuple[float, str | None]:
    if v < lo or v > hi:
        return max(lo, min(hi, v)), f"clamped {v} -> [{lo},{hi}]"
    return v, None
