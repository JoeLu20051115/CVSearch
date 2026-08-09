"""Immutable, answer-independent state scoring contracts."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .types import AnswerRecord


@dataclass(frozen=True)
class _FrozenMapping(Mapping[str, Any]):
    _entries: tuple[tuple[str, Any], ...]

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._entries:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenMapping(tuple((key, _freeze_json(item)) for key, item in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_json(item) for key, item in value._entries}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, init=False)
class ImmutableAnswerSnapshot:
    output: Any
    canonical_answer: Any
    raw_outputs: tuple[Any, ...]
    groups: Mapping[str, Any]
    frequency: float
    margin: float
    confidence: float
    uncertainty: float
    losses: tuple[float, ...]
    selected_from: str
    aggregation_available: bool | None
    aggregation_reason: str | None

    def __init__(self, answer: AnswerRecord) -> None:
        if not isinstance(answer, AnswerRecord):
            raise TypeError("answer must be an AnswerRecord")
        payload = answer.to_dict()
        for name, value in payload.items():
            object.__setattr__(self, name, _freeze_json(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            name: _thaw_json(getattr(self, name))
            for name in (
                "output",
                "canonical_answer",
                "raw_outputs",
                "groups",
                "frequency",
                "margin",
                "confidence",
                "uncertainty",
                "losses",
                "selected_from",
                "aggregation_available",
                "aggregation_reason",
            )
        }


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

    def to_dict(self) -> dict[str, float]:
        return {
            "uncertainty": self.uncertainty,
            "support_avg": self.support_avg,
            "support_min": self.support_min,
            "coverage": self.coverage,
            "normalized_cost": self.normalized_cost,
        }


@dataclass(frozen=True, init=False)
class EvidenceState:
    answer: ImmutableAnswerSnapshot
    features: EvidenceStateScore

    def __init__(self, answer: AnswerRecord, features: EvidenceStateScore) -> None:
        if not isinstance(answer, AnswerRecord):
            raise TypeError("answer must be an AnswerRecord")
        if not isinstance(features, EvidenceStateScore):
            raise TypeError("features must be an EvidenceStateScore")
        object.__setattr__(self, "answer", ImmutableAnswerSnapshot(answer))
        object.__setattr__(self, "features", features)

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer.to_dict(), "features": self.features.to_dict()}


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
