"""Pure label-blind selector for reviewed evidence-gap action states."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


STABILITY_GAIN_THRESHOLD = 0.25
_ACTION_NAME_ORDER = ("EXPAND", "ZOOM")
_P0_FIELDS = frozenset({"action", "output", "p0_stability"})
_CANDIDATE_FIELDS = frozenset({
    "action", "feasible", "output", "candidate_stability",
})
_FORBIDDEN_KEY_FRAGMENTS = (
    "answer", "benchmark", "category", "correct", "evaluator", "groundtruth",
    "label", "ordinal", "question", "resolution", "target", "truth",
)


@dataclass(frozen=True)
class SelectionDecision:
    action: str
    status: str
    output: Any
    stability_gain: float | None


def _snapshot_json(value: Any, active: set[int] | None = None) -> Any:
    """Copy exact JSON builtins without invoking subclass hooks."""
    active = set() if active is None else active
    value_type = type(value)
    if value_type is dict:
        identity = id(value)
        if identity in active:
            raise ValueError("selector DTO must not contain a reference cycle")
        active.add(identity)
        try:
            snapshot = {}
            for key, nested in value.items():
                if type(key) is not str:
                    raise TypeError("selector DTO keys must be exact strings")
                snapshot[key] = _snapshot_json(nested, active)
            return snapshot
        finally:
            active.remove(identity)
    if value_type is list:
        identity = id(value)
        if identity in active:
            raise ValueError("selector DTO must not contain a reference cycle")
        active.add(identity)
        try:
            return [_snapshot_json(nested, active) for nested in value]
        finally:
            active.remove(identity)
    if value is None or value_type in (str, int, bool):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("selector DTO numbers must be finite")
        return value
    raise TypeError("selector DTO must contain only exact JSON builtins")


def _snapshot_inputs(
    p0: Any, candidates: Any,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if type(p0) is not dict:
        raise TypeError("P0 state must be an exact dict")
    if type(candidates) not in (list, tuple):
        raise TypeError("candidates must be an exact list or tuple")
    candidate_values = tuple(candidates)
    if any(type(candidate) is not dict for candidate in candidate_values):
        raise TypeError("candidate state must be an exact dict")
    return (
        _snapshot_json(p0),
        tuple(_snapshot_json(candidate) for candidate in candidate_values),
    )


def _exact_mapping(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be a mapping")
    if set(value) != fields:
        raise ValueError(f"{name} has an invalid exact schema")
    return value


def _reject_forbidden_metadata(value: Any) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            joined_key = re.sub(r"[^a-z0-9]", "", key.casefold())
            if any(fragment in joined_key for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError("selector input contains forbidden metadata")
            _reject_forbidden_metadata(nested)
    elif type(value) is list:
        for nested in value:
            _reject_forbidden_metadata(nested)


def _confidence(value: Any, name: str) -> float:
    stability = _exact_mapping(value, frozenset({"confidence"}), name)
    confidence = stability["confidence"]
    if (
        type(confidence) not in (int, float)
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"{name} confidence must be a finite number in [0, 1]")
    return float(confidence)


def select_unified_state(
    p0: dict[str, Any],
    candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> SelectionDecision:
    p0, candidates = _snapshot_inputs(p0, candidates)
    p0 = _exact_mapping(p0, _P0_FIELDS, "P0 state")
    candidates = tuple(
        _exact_mapping(candidate, _CANDIDATE_FIELDS, "candidate state")
        for candidate in candidates
    )
    _reject_forbidden_metadata(p0)
    for candidate in candidates:
        _reject_forbidden_metadata(candidate)

    if p0["action"] != "P0" or p0["output"] is None:
        raise ValueError("P0 state is not canonical")
    actions = [candidate["action"] for candidate in candidates]
    if any(action not in _ACTION_NAME_ORDER for action in actions):
        raise ValueError("candidate action is not canonical")
    if len(actions) != len(set(actions)):
        raise ValueError("candidate actions must be unique")

    p0_confidence = _confidence(p0["p0_stability"], "P0 stability")
    validated: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        if type(candidate["feasible"]) is not bool:
            raise TypeError("candidate feasible must be a bool")
        if not candidate["feasible"]:
            if candidate["output"] is not None or candidate["candidate_stability"] is not None:
                raise ValueError("infeasible candidate must not contain an output or stability")
            continue
        if candidate["output"] is None:
            raise ValueError("feasible candidate must contain an output")
        gain = _confidence(
            candidate["candidate_stability"], "candidate stability",
        ) - p0_confidence
        validated.append((gain, candidate))

    admitted = [item for item in validated if item[0] >= STABILITY_GAIN_THRESHOLD]
    if admitted:
        gain, selected = min(
            admitted,
            key=lambda item: (-item[0], _ACTION_NAME_ORDER.index(item[1]["action"])),
        )
        return SelectionDecision(
            action=selected["action"],
            status="selected_candidate",
            output=selected["output"],
            stability_gain=gain,
        )
    return SelectionDecision(
        action="P0", status="retained_p0", output=p0["output"],
        stability_gain=None,
    )
