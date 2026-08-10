"""Pure label-blind selector for reviewed evidence-gap action states."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


STABILITY_GAIN_THRESHOLD = 0.25
_ACTION_NAME_ORDER = ("EXPAND", "ZOOM")
_P0_FIELDS = frozenset({"action", "output", "p0_stability"})
_CANDIDATE_FIELDS = frozenset({
    "action", "feasible", "output", "candidate_stability",
})
_FORBIDDEN_FIELD_TOKENS = frozenset({
    "answer", "benchmark", "category", "label", "labels", "ordinal",
    "question", "resolution", "target", "targets",
})


@dataclass(frozen=True)
class SelectionDecision:
    action: str
    status: str
    output: Any
    stability_gain: float | None


def _exact_mapping(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != fields:
        raise ValueError(f"{name} has an invalid exact schema")
    return value


def _reject_forbidden_metadata(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("selector DTO keys must be strings")
            tokens = set(re.findall(r"[a-z0-9]+", key.casefold()))
            if (
                tokens & _FORBIDDEN_FIELD_TOKENS
                or "groundtruth" in tokens
                or {"ground", "truth"} <= tokens
                or {"evaluator", "label"} <= tokens
            ):
                raise ValueError("selector input contains forbidden metadata")
            _reject_forbidden_metadata(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_forbidden_metadata(nested)


def _confidence(value: Any, name: str) -> float:
    stability = _exact_mapping(value, frozenset({"confidence"}), name)
    confidence = stability["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"{name} confidence must be a finite number in [0, 1]")
    return float(confidence)


def select_unified_state(
    p0: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
) -> SelectionDecision:
    p0 = _exact_mapping(p0, _P0_FIELDS, "P0 state")
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")
    candidates = tuple(
        _exact_mapping(candidate, _CANDIDATE_FIELDS, "candidate state")
        for candidate in candidates
    )
    _reject_forbidden_metadata(p0)
    _reject_forbidden_metadata(candidates)

    if p0["action"] != "P0" or p0["output"] is None:
        raise ValueError("P0 state is not canonical")
    actions = [candidate["action"] for candidate in candidates]
    if any(action not in _ACTION_NAME_ORDER for action in actions):
        raise ValueError("candidate action is not canonical")
    if len(actions) != len(set(actions)):
        raise ValueError("candidate actions must be unique")

    p0_confidence = _confidence(p0["p0_stability"], "P0 stability")
    validated: list[tuple[float, Mapping[str, Any]]] = []
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
            output=copy.deepcopy(selected["output"]),
            stability_gain=gain,
        )
    return SelectionDecision(
        action="P0", status="retained_p0", output=copy.deepcopy(p0["output"]),
        stability_gain=None,
    )
