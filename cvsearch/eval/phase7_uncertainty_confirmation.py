"""Pure label-blind admission for deferred search and confirmed actions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


SEARCH_STABILITY_GAIN_THRESHOLD = 0.75
CONFIRMATION_THRESHOLDS = frozenset({0.25, 0.50, 0.75})
_ACTION_NAMES = frozenset({"EXPAND", "ZOOM"})
_P0_FIELDS = frozenset({"action", "output", "p0_stability"})
_CANDIDATE_FIELDS = frozenset({
    "action", "feasible", "output", "stability", "view_sha256",
})
_CONFIRMATION_FIELDS = frozenset({
    "action", "feasible", "output", "stability", "aggregate_stability",
    "view_sha256", "prompt_sha256",
})
_FORBIDDEN_KEY_FRAGMENTS = (
    "answer", "benchmark", "category", "correct", "evaluator",
    "groundtruth", "label", "ordinal", "question", "resolution", "target",
    "truth",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ConfirmationDecision:
    action: str
    status: str
    output: Any
    stability_gain: float | None


def _snapshot_json(value: Any, active: set[int] | None = None) -> Any:
    active = set() if active is None else active
    value_type = type(value)
    if value_type is dict:
        identity = id(value)
        if identity in active:
            raise ValueError("confirmation DTO must not contain a reference cycle")
        active.add(identity)
        try:
            result = {}
            for key, nested in value.items():
                if type(key) is not str:
                    raise TypeError("confirmation DTO keys must be exact strings")
                joined = re.sub(r"[^a-z0-9]", "", key.casefold())
                if any(fragment in joined for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                    raise ValueError("confirmation DTO contains forbidden metadata")
                result[key] = _snapshot_json(nested, active)
            return result
        finally:
            active.remove(identity)
    if value_type is list:
        identity = id(value)
        if identity in active:
            raise ValueError("confirmation DTO must not contain a reference cycle")
        active.add(identity)
        try:
            return [_snapshot_json(item, active) for item in value]
        finally:
            active.remove(identity)
    if value is None or value_type in (str, int, bool):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("confirmation DTO numbers must be finite")
        return value
    raise TypeError("confirmation DTO must contain only exact JSON builtins")


def _exact_mapping(value: Any, fields: frozenset[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    if set(value) != fields:
        raise ValueError(f"{name} has an invalid exact schema")
    return value


def _exact_snapshot(
    value: Any, fields: frozenset[str], name: str,
) -> dict[str, Any]:
    _exact_mapping(value, fields, name)
    return _snapshot_json(value)


def _confidence(value: Any, name: str) -> float:
    stability = _exact_mapping(value, frozenset({"confidence"}), name)
    confidence = stability["confidence"]
    if (
        type(confidence) not in (int, float)
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(f"{name} confidence must be finite and in [0, 1]")
    return float(confidence)


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _p0(value: Any) -> tuple[dict[str, Any], float]:
    p0 = _exact_snapshot(value, _P0_FIELDS, "P0 state")
    if p0["action"] != "P0" or p0["output"] is None:
        raise ValueError("P0 state is not canonical")
    return p0, _confidence(p0["p0_stability"], "P0 stability")


def _candidate(value: Any, allowed_actions: frozenset[str]) -> tuple[dict[str, Any], float | None]:
    candidate = _exact_snapshot(value, _CANDIDATE_FIELDS, "candidate state")
    if candidate["action"] not in allowed_actions:
        raise ValueError("candidate action is not canonical")
    if type(candidate["feasible"]) is not bool:
        raise TypeError("candidate feasible must be an exact bool")
    if not candidate["feasible"]:
        if any(candidate[key] is not None for key in ("output", "stability", "view_sha256")):
            raise ValueError("infeasible candidate must have null observation fields")
        return candidate, None
    if candidate["output"] is None:
        raise ValueError("feasible candidate must contain an output")
    _sha256(candidate["view_sha256"], "candidate view")
    return candidate, _confidence(candidate["stability"], "candidate stability")


def _confirmation(value: Any, action: str) -> tuple[dict[str, Any], float | None]:
    confirmation = _exact_snapshot(
        value, _CONFIRMATION_FIELDS, "confirmation state",
    )
    if confirmation["action"] != action:
        raise ValueError("confirmation action differs from its candidate")
    if type(confirmation["feasible"]) is not bool:
        raise TypeError("confirmation feasible must be an exact bool")
    if not confirmation["feasible"]:
        if any(confirmation[key] is not None for key in (
            "output", "stability", "aggregate_stability", "view_sha256",
            "prompt_sha256",
        )):
            raise ValueError("infeasible confirmation must have null observation fields")
        return confirmation, None
    if confirmation["output"] is None:
        raise ValueError("feasible confirmation must contain an output")
    _confidence(confirmation["stability"], "confirmation stability")
    aggregate = _confidence(
        confirmation["aggregate_stability"], "aggregate stability",
    )
    _sha256(confirmation["view_sha256"], "confirmation view")
    prompts = confirmation["prompt_sha256"]
    if type(prompts) is not list or not prompts:
        raise ValueError("confirmation prompt hashes must be a nonempty list")
    for index, prompt in enumerate(prompts):
        _sha256(prompt, f"confirmation prompt {index}")
    if len(set(prompts)) != len(prompts):
        raise ValueError("confirmation prompt hashes must be distinct")
    return confirmation, aggregate


def _retained(p0: dict[str, Any]) -> ConfirmationDecision:
    return ConfirmationDecision(
        action="P0", status="retained_p0", output=p0["output"],
        stability_gain=None,
    )


def confirm_search_candidate(p0: dict[str, Any], candidate: dict[str, Any]) -> ConfirmationDecision:
    """Admit a paired gate-0.8 SEARCH state only at the frozen high-gain boundary."""
    p0, p0_confidence = _p0(p0)
    candidate, candidate_confidence = _candidate(candidate, frozenset({"SEARCH"}))
    if not candidate["feasible"]:
        return _retained(p0)
    if candidate["output"] == p0["output"]:
        return _retained(p0)
    if candidate_confidence is None:
        raise AssertionError("feasible candidate lost its stability")
    gain = candidate_confidence - p0_confidence
    if gain < SEARCH_STABILITY_GAIN_THRESHOLD:
        return _retained(p0)
    return ConfirmationDecision(
        action="SEARCH", status="selected_deferred_search",
        output=candidate["output"], stability_gain=gain,
    )


def confirm_action_candidate(
    p0: dict[str, Any], candidate: dict[str, Any], confirmation: dict[str, Any],
    *, threshold: float,
) -> ConfirmationDecision:
    """Admit one uncertain action only after a distinct view confirms it."""
    if type(threshold) not in (int, float) or float(threshold) not in CONFIRMATION_THRESHOLDS:
        raise ValueError("confirmation threshold must be one frozen coarse value")
    threshold = float(threshold)
    p0, p0_confidence = _p0(p0)
    candidate, _ = _candidate(candidate, _ACTION_NAMES)
    confirmation, aggregate_confidence = _confirmation(
        confirmation, candidate["action"],
    )
    if not candidate["feasible"] or not confirmation["feasible"]:
        return _retained(p0)
    if candidate["output"] == p0["output"]:
        return _retained(p0)
    if candidate["output"] != confirmation["output"]:
        return _retained(p0)
    if candidate["view_sha256"] == confirmation["view_sha256"]:
        return _retained(p0)
    if aggregate_confidence is None:
        raise AssertionError("feasible confirmation lost aggregate stability")
    gain = aggregate_confidence - p0_confidence
    if aggregate_confidence < threshold or gain <= 0.0:
        return _retained(p0)
    return ConfirmationDecision(
        action=candidate["action"], status="selected_confirmed_action",
        output=confirmation["output"], stability_gain=gain,
    )
