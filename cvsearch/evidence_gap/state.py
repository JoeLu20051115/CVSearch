"""Immutable, answer-independent state scoring contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .types import AnswerRecord


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or type(value).__name__ == "bool" or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _nonnegative_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or type(value).__name__ == "bool" or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


@dataclass(frozen=True)
class EvidenceStateScore:
    uncertainty: float
    support_avg: float
    support_min: float
    coverage: float
    normalized_cost: float

    def __post_init__(self) -> None:
        for name in ("uncertainty", "support_avg", "support_min", "coverage", "normalized_cost"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))


@dataclass(frozen=True)
class EvidenceState:
    answer: AnswerRecord
    features: EvidenceStateScore

    def __post_init__(self) -> None:
        if not isinstance(self.answer, AnswerRecord):
            raise TypeError("answer must be an AnswerRecord")
        if not isinstance(self.features, EvidenceStateScore):
            raise TypeError("features must be an EvidenceStateScore")


def score_state(state: EvidenceStateScore) -> float:
    """Apply the frozen, benchmark-independent history score."""
    if not isinstance(state, EvidenceStateScore):
        raise TypeError("state must be an EvidenceStateScore")
    return (
        0.35 * (1.0 - state.uncertainty)
        + 0.25 * state.support_min
        + 0.15 * state.support_avg
        + 0.25 * state.coverage
        - 0.10 * state.normalized_cost
    )


def select_state(anchor: EvidenceState, candidate: EvidenceState, tau: Any) -> tuple[EvidenceState, float]:
    """Replace only on a strictly positive global evidence margin."""
    if not isinstance(anchor, EvidenceState) or not isinstance(candidate, EvidenceState):
        raise TypeError("anchor and candidate must be EvidenceState instances")
    threshold = _nonnegative_finite(tau, "tau")
    margin = score_state(candidate.features) - score_state(anchor.features)
    return (candidate, margin) if margin > threshold else (anchor, margin)
