"""Pure, feature-gated evidence-search stop and fallback policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from .types import BACKTRACK, EXPAND, FORCED_RETURN, NEXT, SPLIT, ZOOM, AnswerRecord, HistoryRecord


_OBSERVATION_ACTIONS = (ZOOM, SPLIT, EXPAND, NEXT, BACKTRACK)


def _finite(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if type(value).__module__ == "numpy" and type(value).__name__ == "bool":
        return bool(value)
    raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class StopThresholds:
    """Strict certification thresholds; each is a finite probability."""

    gap: float = 0.2
    uncertainty: float = 0.2
    support_avg: float = 0.7
    support_min: float = 0.6

    def __post_init__(self) -> None:
        for name in ("gap", "uncertainty", "support_avg", "support_min"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, minimum=0.0, maximum=1.0))


@dataclass(frozen=True)
class PolicyState:
    """Minimal action-feasibility state; advanced observations default off."""

    budget_exhausted: bool = False
    has_unvisited_next: bool = False
    has_unvisited_history_branch: bool = False
    backtrack_requested: bool = False
    failed_actions: tuple[str, ...] = ()
    progress_deltas: tuple[float, ...] = ()
    progress_threshold: float = 0.03
    zoom_available: bool = False
    zoom_enabled: bool = False
    split_available: bool = False
    split_enabled: bool = False
    expand_available: bool = False
    expand_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "budget_exhausted", "has_unvisited_next", "has_unvisited_history_branch",
            "backtrack_requested", "zoom_available", "zoom_enabled", "split_available",
            "split_enabled", "expand_available", "expand_enabled",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        actions = tuple(self.failed_actions)
        if len(actions) != len(set(actions)):
            raise ValueError("failed_actions must not contain duplicates")
        if any(not isinstance(action, str) or action not in _OBSERVATION_ACTIONS for action in actions):
            raise ValueError("failed_actions contains an unknown or invalid action")
        object.__setattr__(self, "failed_actions", actions)
        deltas = tuple(_finite(delta, "progress delta", minimum=0.0) for delta in self.progress_deltas)
        object.__setattr__(self, "progress_deltas", deltas)
        object.__setattr__(self, "progress_threshold", _finite(
            self.progress_threshold, "progress_threshold", minimum=0.0))


def feasible_actions(state: PolicyState) -> tuple[str, ...]:
    """Return ordered actions that can still change the search state."""
    if not isinstance(state, PolicyState):
        raise TypeError("state must be a PolicyState")
    if state.budget_exhausted:
        return (FORCED_RETURN,)

    failed = set(state.failed_actions)
    actions: list[str] = []
    for action, available, enabled in (
        (ZOOM, state.zoom_available, state.zoom_enabled),
        (SPLIT, state.split_available, state.split_enabled),
        (EXPAND, state.expand_available, state.expand_enabled),
    ):
        if available and enabled and action not in failed:
            actions.append(action)
    if state.has_unvisited_next and NEXT not in failed:
        actions.append(NEXT)
    stalled = (
        len(state.progress_deltas) >= 2
        and all(delta < state.progress_threshold for delta in state.progress_deltas[-2:])
    )
    if (state.backtrack_requested or stalled) and state.has_unvisited_history_branch and BACKTRACK not in failed:
        actions.append(BACKTRACK)
    return tuple(actions) or (FORCED_RETURN,)


def should_certify(gaps: Mapping[str, Any], answer: AnswerRecord, support_avg: float,
                   support_min: float, thresholds: StopThresholds) -> bool:
    """Apply all four evidence gates; empty gaps deliberately cannot certify."""
    if not isinstance(gaps, Mapping) or not gaps:
        raise ValueError("gaps must be a non-empty mapping")
    if not isinstance(answer, AnswerRecord):
        raise TypeError("answer must be an AnswerRecord")
    if not isinstance(thresholds, StopThresholds):
        raise TypeError("thresholds must be StopThresholds")
    values = []
    for key, value in gaps.items():
        if not isinstance(key, str):
            raise TypeError("gap keys must be strings")
        values.append(_finite(value, f"gap[{key!r}]", minimum=0.0, maximum=1.0))
    uncertainty = _finite(answer.uncertainty, "answer.uncertainty", minimum=0.0, maximum=1.0)
    average = _finite(support_avg, "support_avg", minimum=0.0, maximum=1.0)
    minimum = _finite(support_min, "support_min", minimum=0.0, maximum=1.0)
    return (
        max(values) < thresholds.gap
        and uncertainty < thresholds.uncertainty
        and average >= thresholds.support_avg
        and minimum >= thresholds.support_min
    )


class HistoryBuffer:
    """JSON-safe snapshots, rejecting duplicate steps to preserve tie ordering."""

    def __init__(self) -> None:
        self._records: list[HistoryRecord] = []
        self._steps: set[float] = set()

    @staticmethod
    def _validate(record: HistoryRecord) -> None:
        if not isinstance(record, HistoryRecord):
            raise TypeError("record must be a HistoryRecord")
        _finite(record.step, "record.step", minimum=0.0)
        if not isinstance(record.answer, AnswerRecord):
            raise TypeError("record.answer must be an AnswerRecord")
        _finite(record.answer.confidence, "record.answer.confidence", minimum=0.0, maximum=1.0)
        _finite(record.support_avg, "record.support_avg", minimum=0.0, maximum=1.0)
        _finite(record.support_min, "record.support_min", minimum=0.0, maximum=1.0)
        _finite(record.cost, "record.cost", minimum=0.0)
        _bool(record.has_unvisited_branch, "record.has_unvisited_branch")
        record.to_dict()

    def add(self, record: HistoryRecord) -> None:
        self._validate(record)
        step = float(record.step)
        if step in self._steps:
            raise ValueError(f"duplicate history step: {record.step}")
        snapshot = copy.deepcopy(record)
        self._records.append(snapshot)
        self._steps.add(step)

    @staticmethod
    def _rank(record: HistoryRecord) -> tuple[float, float, float, float, float]:
        return (-float(record.answer.confidence), -float(record.support_min),
                -float(record.support_avg), float(record.cost), float(record.step))

    def _best(self, records: list[HistoryRecord]) -> HistoryRecord | None:
        return None if not records else copy.deepcopy(min(records, key=self._rank))

    def best(self) -> HistoryRecord | None:
        return self._best(self._records)

    def best_with_unvisited_branch(self) -> HistoryRecord | None:
        return self._best([record for record in self._records if record.has_unvisited_branch])


def select_root_or_search(root: AnswerRecord, search: AnswerRecord, tolerance: float) -> AnswerRecord:
    """Choose root only when it is materially more confident than search."""
    if not isinstance(root, AnswerRecord) or not isinstance(search, AnswerRecord):
        raise TypeError("root and search must be AnswerRecord instances")
    root_confidence = _finite(root.confidence, "root.confidence", minimum=0.0, maximum=1.0)
    search_confidence = _finite(search.confidence, "search.confidence", minimum=0.0, maximum=1.0)
    allowed_difference = _finite(tolerance, "tolerance", minimum=0.0)
    if search_confidence + allowed_difference < root_confidence:
        return replace(root, selected_from="root")
    return replace(search, selected_from="search")
