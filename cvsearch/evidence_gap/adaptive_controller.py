"""Model-neutral calibration and action policy after frozen candidate ranking."""

from __future__ import annotations

import bisect
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .types import BACKTRACK, EXPAND, SPLIT, ZOOM


STOP = "STOP"
_ACTIONS = (ZOOM, EXPAND, SPLIT, BACKTRACK, STOP)
_MISSING_REASONS = frozenset({
    "none", "detail_unreadable", "context_missing", "location_ambiguous",
    "target_missing", "conflict",
})
_FORBIDDEN_KEY_FRAGMENTS = (
    "answer", "benchmark", "category", "correct", "evaluator", "groundtruth",
    "label", "ordinal", "resolution", "targetbox", "truth",
)


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _reject_forbidden_metadata(value: Any) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError("evidence requirement keys must be exact strings")
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ValueError("evidence requirements contain evaluator metadata")
            _reject_forbidden_metadata(nested)
    elif type(value) is list:
        for nested in value:
            _reject_forbidden_metadata(nested)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise TypeError("evidence requirements must contain exact JSON builtins")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("evidence requirements must contain finite numbers")


@dataclass(frozen=True)
class EvidenceDemand:
    detail: float
    context: float
    localization: float

    def __post_init__(self) -> None:
        for name in ("detail", "context", "localization"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))

    def action_prior(self) -> dict[str, float]:
        weights = {
            ZOOM: self.detail,
            EXPAND: self.context,
            SPLIT: self.localization,
        }
        total = sum(weights.values())
        if total == 0.0:
            return {ZOOM: 1 / 3, EXPAND: 1 / 3, SPLIT: 1 / 3}
        return {name: value / total for name, value in weights.items()}

    def to_dict(self) -> dict[str, float]:
        return {
            "detail": self.detail,
            "context": self.context,
            "localization": self.localization,
        }


def infer_evidence_demand(requirements: Sequence[Mapping[str, Any]]) -> EvidenceDemand:
    """Infer a bounded action prior without using benchmark or evaluator fields."""
    if isinstance(requirements, (str, bytes)) or not isinstance(requirements, Sequence):
        raise TypeError("requirements must be a sequence")
    snapshot = tuple(requirements)
    for item in snapshot:
        if type(item) is not dict:
            raise TypeError("each evidence requirement must be an exact dictionary")
        _reject_forbidden_metadata(item)

    detail = 0.0
    context = 0.0
    localization = 0.0
    target_details = 0
    for item in snapshot:
        kind = item.get("kind")
        if kind == "target_detail":
            detail = 1.0
            localization = max(localization, 0.25)
            target_details += 1
        elif kind == "relation_context":
            context = 1.0
            localization = max(localization, 0.5)
        elif kind == "coverage":
            context = 1.0
        elif kind == "question_evidence":
            detail = max(detail, 0.5)
            localization = max(localization, 0.25)
        elif kind == "runtime_ranking_context":
            continue
        else:
            raise ValueError("unknown evidence requirement kind")
    if target_details > 1:
        context = max(context, 0.5)
        localization = max(localization, 0.5)
    return EvidenceDemand(detail, context, localization)


@dataclass(frozen=True)
class SupportObservation:
    p_full: float
    p_partial: float
    p_none: float
    support_consistency: float
    answer_consistency: float
    missing_reason: str
    normalized_cost: float

    def __post_init__(self) -> None:
        for name in (
            "p_full", "p_partial", "p_none", "support_consistency",
            "answer_consistency", "normalized_cost",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if abs(self.p_full + self.p_partial + self.p_none - 1.0) > 1e-9:
            raise ValueError("support probabilities must sum to one")
        if type(self.missing_reason) is not str or self.missing_reason not in _MISSING_REASONS:
            raise ValueError("missing_reason is not canonical")

    @property
    def raw_support(self) -> float:
        return self.p_full * math.sqrt(
            self.support_consistency * self.answer_consistency
        )

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.raw_support

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_full": self.p_full,
            "p_partial": self.p_partial,
            "p_none": self.p_none,
            "support_consistency": self.support_consistency,
            "answer_consistency": self.answer_consistency,
            "missing_reason": self.missing_reason,
            "normalized_cost": self.normalized_cost,
            "raw_support": self.raw_support,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class IsotonicCalibrator:
    upper_bounds: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.upper_bounds or len(self.upper_bounds) != len(self.probabilities):
            raise ValueError("isotonic calibrator requires aligned nonempty knots")
        bounds = tuple(_unit(value, "isotonic upper bound") for value in self.upper_bounds)
        probabilities = tuple(
            _unit(value, "isotonic probability") for value in self.probabilities
        )
        if any(left >= right for left, right in zip(bounds, bounds[1:])):
            raise ValueError("isotonic upper bounds must be strictly increasing")
        if any(left > right for left, right in zip(probabilities, probabilities[1:])):
            raise ValueError("isotonic probabilities must be nondecreasing")
        object.__setattr__(self, "upper_bounds", bounds)
        object.__setattr__(self, "probabilities", probabilities)

    def predict(self, raw_support: Any) -> float:
        value = _unit(raw_support, "raw_support")
        index = min(bisect.bisect_left(self.upper_bounds, value), len(self.probabilities) - 1)
        return self.probabilities[index]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "upper_bounds": list(self.upper_bounds),
            "probabilities": list(self.probabilities),
        }


def fit_isotonic(samples: Sequence[tuple[float, int]]) -> IsotonicCalibrator:
    """Fit deterministic binary isotonic calibration with the PAVA algorithm."""
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("calibration samples must be a sequence")
    validated: list[tuple[float, int]] = []
    for sample in samples:
        if type(sample) is not tuple or len(sample) != 2:
            raise TypeError("each calibration sample must be an exact pair")
        score, label = sample
        score = _unit(score, "calibration score")
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("calibration labels must be exact binary integers")
        validated.append((score, label))
    if not validated:
        raise ValueError("calibration samples must not be empty")

    grouped: list[list[float]] = []
    for score, label in sorted(validated):
        if grouped and grouped[-1][0] == score:
            grouped[-1][1] += label
            grouped[-1][2] += 1
        else:
            grouped.append([score, float(label), 1])

    blocks: list[list[float]] = []
    for upper, positive, count in grouped:
        blocks.append([upper, positive, count])
        while (
            len(blocks) >= 2
            and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([
                right[0], left[1] + right[1], left[2] + right[2],
            ])
    return IsotonicCalibrator(
        tuple(block[0] for block in blocks),
        tuple(block[1] / block[2] for block in blocks),
    )


@dataclass(frozen=True)
class ActionDecision:
    action: str
    reason: str
    calibrated_support: float
    uncertainty: float
    prior_weight: float
    scores: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ValueError("decision action is not canonical")
        if type(self.reason) is not str or not self.reason:
            raise ValueError("decision reason must be nonempty")
        for name in ("calibrated_support", "uncertainty", "prior_weight"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if abs(self.calibrated_support + self.uncertainty - 1.0) > 1e-12:
            raise ValueError("decision uncertainty must complement calibrated support")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "calibrated_support": self.calibrated_support,
            "uncertainty": self.uncertainty,
            "prior_weight": self.prior_weight,
            "scores": {name: value for name, value in self.scores},
        }


def select_adaptive_action(
    demand: EvidenceDemand,
    observation: SupportObservation,
    *,
    available: Sequence[str],
    trajectory: Sequence[float],
    has_unvisited_branch: bool,
    calibrated_support: float | None = None,
    step: int = 0,
    action_costs: Mapping[str, float] | None = None,
    stop_threshold: float = 0.85,
    stall_threshold: float = 0.03,
) -> ActionDecision:
    """Choose one reversible action; observed support overrides the soft prior."""
    if not isinstance(demand, EvidenceDemand) or not isinstance(observation, SupportObservation):
        raise TypeError("demand and observation must use normalized controller DTOs")
    if isinstance(available, (str, bytes)) or not isinstance(available, Sequence):
        raise TypeError("available actions must be a sequence")
    actions = tuple(available)
    if not actions or len(actions) != len(set(actions)) or any(
        type(action) is not str or action not in _ACTIONS for action in actions
    ):
        raise ValueError("available actions must be unique canonical actions")
    if type(has_unvisited_branch) is not bool:
        raise TypeError("has_unvisited_branch must be boolean")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative exact integer")
    support = observation.raw_support if calibrated_support is None else _unit(
        calibrated_support, "calibrated_support"
    )
    stop_threshold = _unit(stop_threshold, "stop_threshold")
    stall_threshold = _unit(stall_threshold, "stall_threshold")
    progress = tuple(_unit(value, "trajectory support") for value in trajectory)
    prior_weight = 0.3 / (step + 1)

    def decision(action: str, reason: str, scores: tuple[tuple[str, float], ...] = ()):
        return ActionDecision(
            action=action,
            reason=reason,
            calibrated_support=support,
            uncertainty=1.0 - support,
            prior_weight=prior_weight,
            scores=scores,
        )

    stable = observation.support_consistency >= 0.8 and observation.answer_consistency >= 0.8
    if STOP in actions and support >= stop_threshold and stable and observation.missing_reason == "none":
        return decision(STOP, "calibrated_support_sufficient")

    stalled = (
        len(progress) >= 3
        and progress[-1] - progress[-2] < stall_threshold
        and progress[-2] - progress[-3] < stall_threshold
    )
    if stalled and has_unvisited_branch and BACKTRACK in actions:
        return decision(BACKTRACK, "support_trajectory_stalled")

    requested = {
        "detail_unreadable": ZOOM,
        "context_missing": EXPAND,
        "location_ambiguous": SPLIT,
    }.get(observation.missing_reason)
    if requested in actions:
        return decision(requested, f"observed_{observation.missing_reason}")
    if (
        observation.missing_reason in {"target_missing", "conflict"}
        and has_unvisited_branch
        and BACKTRACK in actions
    ):
        return decision(BACKTRACK, f"observed_{observation.missing_reason}")

    costs = {} if action_costs is None else dict(action_costs)
    if any(action not in _ACTIONS for action in costs):
        raise ValueError("action costs contain an unknown action")
    normalized_costs = {action: _unit(value, f"cost[{action}]") for action, value in costs.items()}
    prior = demand.action_prior()
    candidates = tuple(
        action for action in actions
        if action != STOP and (action != BACKTRACK or has_unvisited_branch)
    )
    if not candidates:
        return decision(actions[0], "only_available_action")
    scored = tuple((
        action,
        prior_weight * prior.get(action, 0.0) - 0.2 * normalized_costs.get(action, 0.0),
    ) for action in candidates)
    best = min(scored, key=lambda item: (-item[1], _ACTIONS.index(item[0])))
    return decision(best[0], "soft_prior_cost_tiebreak", scored)


__all__ = [
    "BACKTRACK", "EXPAND", "SPLIT", "STOP", "ZOOM", "ActionDecision",
    "EvidenceDemand", "IsotonicCalibrator", "SupportObservation", "fit_isotonic",
    "infer_evidence_demand", "select_adaptive_action",
]
