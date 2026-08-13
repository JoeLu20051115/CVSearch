#!/usr/bin/env python3
"""Label-blind replay primitives for the uncertainty-support selector."""

from __future__ import annotations

import copy
import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple, dataclass
from typing import Any

from cvsearch.eval.replay_adaptive_search import (
    FrozenCalibration,
    FrozenSelectedCalibration,
    _answer_record,
    _canonical_json,
    _p0_canonical_answer,
    _rank_digest,
    replay_adaptive_search,
)
from cvsearch.eval.replay_split_search import (
    _candidate_output,
    _sha256,
    _split_audit,
    _stage2_support,
    _unit,
)


PROFILES: Mapping[str, tuple[float, ...]] = {
    "balanced": (0.20, 0.20, 0.20, 0.20, 0.20),
    "support_heavy": (0.10, 0.15, 0.35, 0.20, 0.20),
    "uncertainty_light": (0.10, 0.25, 0.25, 0.20, 0.20),
}
THRESHOLDS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
_REPLAY_ROW_FIELDS = frozenset({
    "_eg_ordinal", "answer_type", "options", "output", "method_trace",
})


def sanitize_replay_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a raw evaluator row onto the exact label-free replay schema."""
    if type(row) is not dict:
        raise TypeError("selector input rows must be exact dictionaries")
    result = {
        key: copy.deepcopy(row[key])
        for key in _REPLAY_ROW_FIELDS if key in row
    }
    if set(result) != _REPLAY_ROW_FIELDS:
        raise ValueError("selector input row lacks a required inference field")
    return result


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class AdvantageFeatures:
    """The five normalized candidate-vs-P0 inputs to the unified score."""

    uncertainty: float
    agreement: float
    support: float
    support_gain_01: float
    conflict_margin_01: float

    def __post_init__(self) -> None:
        for name, value in zip(self.__dataclass_fields__, astuple(self)):
            object.__setattr__(self, name, _unit_float(value, name))


@dataclass(frozen=True)
class AggregateEvidenceFeatures:
    """Label-blind evidence aggregates available at one replay checkpoint."""

    uncertainty: float
    agreement: float
    minimum_support: float
    support_gain_01: float
    conflict_margin_01: float
    mean_support: float
    maximum_support: float
    support_dispersion: float
    agreeing_fraction: float
    parseable_fraction: float
    agreeing_branch_fraction: float
    agreeing_role_fraction: float
    recent_agreement: float
    observation_fraction: float

    def __post_init__(self) -> None:
        for name, value in zip(self.__dataclass_fields__, astuple(self)):
            object.__setattr__(self, name, _unit_float(value, name))


def raw_advantage(
    features: AdvantageFeatures,
    weights: Sequence[float],
) -> float:
    """Return the sole weighted numeric score used by the selector."""
    if not isinstance(features, AdvantageFeatures):
        raise TypeError("features must be AdvantageFeatures")
    if len(weights) != 5:
        raise ValueError("weight profile must contain five values")
    normalized = tuple(_unit_float(value, "weight") for value in weights)
    if not math.isclose(math.fsum(normalized), 1.0, abs_tol=1e-12):
        raise ValueError("weight profile must sum to one")
    return math.fsum(
        weight * value for weight, value in zip(normalized, astuple(features))
    )


@dataclass(frozen=True)
class UtilityIsotonicCalibrator:
    """A deterministic monotone mapping from raw score to relative utility."""

    upper_bounds: tuple[float, ...]
    utilities: tuple[float, ...]

    def _validate(self) -> None:
        if not self.upper_bounds or len(self.upper_bounds) != len(self.utilities):
            raise ValueError("utility calibrator arrays must be nonempty and aligned")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.upper_bounds
        ):
            raise ValueError("utility calibrator bounds must be finite")
        if any(
            left >= right
            for left, right in zip(self.upper_bounds, self.upper_bounds[1:])
        ):
            raise ValueError("utility calibrator bounds must be strictly increasing")
        values = tuple(_unit_float(value, "utility") for value in self.utilities)
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("utility calibrator values must be nondecreasing")

    def predict(self, score: float) -> float:
        self._validate()
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("utility score must be numeric")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("utility score must be finite")
        index = min(bisect_left(self.upper_bounds, value), len(self.utilities) - 1)
        return float(self.utilities[index])

    def to_dict(self) -> dict[str, list[float]]:
        self._validate()
        return {
            "upper_bounds": [float(value) for value in self.upper_bounds],
            "utilities": [float(value) for value in self.utilities],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UtilityIsotonicCalibrator:
        if not isinstance(payload, Mapping):
            raise TypeError("utility calibrator payload must be a mapping")
        bounds = payload.get("upper_bounds")
        utilities = payload.get("utilities")
        if not isinstance(bounds, list) or not isinstance(utilities, list):
            raise ValueError("utility calibrator payload arrays are missing")
        result = cls(tuple(bounds), tuple(utilities))
        result._validate()
        return result


def fit_utility_isotonic(
    samples: Sequence[tuple[float, float]],
) -> UtilityIsotonicCalibrator:
    """Fit continuous-target isotonic regression with deterministic PAVA."""
    if not samples:
        raise ValueError("utility calibration samples must be nonempty")
    validated = []
    for score, target in samples:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("utility calibration score must be numeric")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("utility calibration score must be finite")
        validated.append((score, _unit_float(target, "utility target")))
    validated.sort()

    grouped: list[tuple[float, float, int]] = []
    for score, target in validated:
        if grouped and grouped[-1][0] == score:
            bound, total, count = grouped[-1]
            grouped[-1] = (bound, total + target, count + 1)
        else:
            grouped.append((score, target, 1))

    blocks: list[dict[str, float | int]] = []
    for index, (_, total, count) in enumerate(grouped):
        blocks.append({"start": index, "end": index, "total": total, "count": count})
        while len(blocks) >= 2:
            left, right = blocks[-2:]
            left_mean = float(left["total"]) / int(left["count"])
            right_mean = float(right["total"]) / int(right["count"])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [{
                "start": int(left["start"]),
                "end": int(right["end"]),
                "total": float(left["total"]) + float(right["total"]),
                "count": int(left["count"]) + int(right["count"]),
            }]

    utilities = [0.0] * len(grouped)
    for block in blocks:
        mean = float(block["total"]) / int(block["count"])
        for index in range(int(block["start"]), int(block["end"]) + 1):
            utilities[index] = mean
    compact_bounds: list[float] = []
    compact_utilities: list[float] = []
    for item, utility in zip(grouped, utilities):
        if compact_utilities and utility == compact_utilities[-1]:
            compact_bounds[-1] = item[0]
        else:
            compact_bounds.append(item[0])
            compact_utilities.append(utility)
    result = UtilityIsotonicCalibrator(
        tuple(compact_bounds),
        tuple(compact_utilities),
    )
    result._validate()
    return result


def risk_basis(features: AdvantageFeatures) -> tuple[float, ...]:
    """Expand the five evidence values into the frozen quadratic risk basis."""
    if not isinstance(features, AdvantageFeatures):
        raise TypeError("risk features must be AdvantageFeatures")
    values = astuple(features)
    return (
        1.0,
        *values,
        *(value * value for value in values),
        *(values[0] * value for value in values[1:]),
        *(values[1] * value for value in values[2:]),
    )


@dataclass(frozen=True)
class RiskLinearHead:
    """One deterministic linear head over the frozen quadratic basis."""

    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.coefficients) != 18:
            raise ValueError("risk head must contain eighteen coefficients")
        normalized = []
        for value in self.coefficients:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("risk coefficients must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("risk coefficients must be finite")
            normalized.append(value)
        object.__setattr__(self, "coefficients", tuple(normalized))

    def predict(self, features: AdvantageFeatures) -> float:
        value = math.fsum(
            coefficient * basis
            for coefficient, basis in zip(
                self.coefficients, risk_basis(features),
            )
        )
        return min(1.0, max(0.0, value))

    def to_dict(self) -> dict[str, list[float]]:
        return {"coefficients": list(self.coefficients)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RiskLinearHead:
        if not isinstance(payload, Mapping):
            raise TypeError("risk head payload must be a mapping")
        values = payload.get("coefficients")
        if not isinstance(values, list):
            raise ValueError("risk head coefficients are missing")
        return cls(tuple(values))


@dataclass(frozen=True)
class RiskLogisticHead:
    """One standardized logistic head over aggregate evidence features."""

    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        width = len(AggregateEvidenceFeatures.__dataclass_fields__)
        if not all(
            len(values) == width
            for values in (self.coefficients, self.means, self.scales)
        ):
            raise ValueError("logistic risk head arrays must match evidence width")
        for name in ("intercept",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        normalized = []
        for name, values in (
            ("coefficients", self.coefficients),
            ("means", self.means),
            ("scales", self.scales),
        ):
            current = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"logistic risk {name} must be numeric")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"logistic risk {name} must be finite")
                if name == "scales" and value <= 0.0:
                    raise ValueError("logistic risk scales must be positive")
                current.append(value)
            normalized.append(tuple(current))
        object.__setattr__(self, "coefficients", normalized[0])
        object.__setattr__(self, "means", normalized[1])
        object.__setattr__(self, "scales", normalized[2])

    def predict(self, features: AggregateEvidenceFeatures) -> float:
        if not isinstance(features, AggregateEvidenceFeatures):
            raise TypeError("logistic risk features must be aggregate evidence")
        logit = self.intercept + math.fsum(
            coefficient * ((value - mean) / scale)
            for coefficient, value, mean, scale in zip(
                self.coefficients, astuple(features), self.means, self.scales,
            )
        )
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        exponential = math.exp(logit)
        return exponential / (1.0 + exponential)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "means": list(self.means),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RiskLogisticHead:
        if not isinstance(payload, Mapping):
            raise TypeError("logistic risk head payload must be a mapping")
        arrays = tuple(payload.get(name) for name in (
            "coefficients", "means", "scales",
        ))
        if any(not isinstance(values, list) for values in arrays):
            raise ValueError("logistic risk head arrays are missing")
        return cls(payload.get("intercept"), *(tuple(values) for values in arrays))


@dataclass(frozen=True)
class RiskRegion:
    """One axis-aligned leaf in the calibrated five-feature risk state."""

    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    expected_benefit: float
    corruption_risk: float
    margin: float

    def __post_init__(self) -> None:
        if len(self.lower_bounds) != 5 or len(self.upper_bounds) != 5:
            raise ValueError("risk region bounds must contain five values")
        lower = tuple(
            _unit_float(value, "risk region lower bound")
            for value in self.lower_bounds
        )
        upper = tuple(
            _unit_float(value, "risk region upper bound")
            for value in self.upper_bounds
        )
        if any(left > right for left, right in zip(lower, upper)):
            raise ValueError("risk region lower bound exceeds upper bound")
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(
            self, "expected_benefit",
            _unit_float(self.expected_benefit, "region expected benefit"),
        )
        object.__setattr__(
            self, "corruption_risk",
            _unit_float(self.corruption_risk, "region corruption risk"),
        )
        if isinstance(self.margin, bool) or not isinstance(
            self.margin, (int, float),
        ):
            raise TypeError("risk region margin must be numeric")
        margin = float(self.margin)
        if not math.isfinite(margin):
            raise ValueError("risk region margin must be finite")
        object.__setattr__(self, "margin", margin)

    def matches(self, features: AdvantageFeatures) -> bool:
        return all(
            lower <= value <= upper
            for lower, value, upper in zip(
                self.lower_bounds, astuple(features), self.upper_bounds,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "expected_benefit": self.expected_benefit,
            "corruption_risk": self.corruption_risk,
            "margin": self.margin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RiskRegion:
        if not isinstance(payload, Mapping):
            raise TypeError("risk region payload must be a mapping")
        lower = payload.get("lower_bounds")
        upper = payload.get("upper_bounds")
        if not isinstance(lower, list) or not isinstance(upper, list):
            raise ValueError("risk region bounds are missing")
        return cls(
            tuple(lower), tuple(upper), payload.get("expected_benefit"),
            payload.get("corruption_risk"), payload.get("margin"),
        )


@dataclass(frozen=True)
class RiskCalibrator:
    """Benefit and harm heads reduced to one risk-adjusted decision margin."""

    benefit_head: RiskLinearHead
    harm_head: RiskLinearHead
    risk_penalty: float
    decision_boundary: float
    regions: tuple[RiskRegion, ...] = ()
    evidence_benefit_head: RiskLogisticHead | None = None
    evidence_harm_head: RiskLogisticHead | None = None
    minimum_observations: int = 1
    maximum_observations: int = 14
    minimum_agreeing_views: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.benefit_head, RiskLinearHead):
            raise TypeError("benefit_head must be a frozen risk head")
        if not isinstance(self.harm_head, RiskLinearHead):
            raise TypeError("harm_head must be a frozen risk head")
        for name in ("risk_penalty", "decision_boundary"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.risk_penalty < 0.0:
            raise ValueError("risk penalty must be nonnegative")
        if any(not isinstance(region, RiskRegion) for region in self.regions):
            raise TypeError("risk regions must be frozen RiskRegion values")
        evidence_heads = (self.evidence_benefit_head, self.evidence_harm_head)
        if (evidence_heads[0] is None) != (evidence_heads[1] is None):
            raise ValueError("aggregate benefit and harm heads must be paired")
        if any(
            head is not None and not isinstance(head, RiskLogisticHead)
            for head in evidence_heads
        ):
            raise TypeError("aggregate risk heads must be logistic heads")
        for name in (
            "minimum_observations", "maximum_observations",
            "minimum_agreeing_views",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if self.minimum_observations > self.maximum_observations:
            raise ValueError("minimum observations exceed maximum observations")
        if self.maximum_observations > 14:
            raise ValueError("maximum observations exceed the frozen search budget")
        if self.minimum_agreeing_views > 14:
            raise ValueError("minimum agreeing views exceed the frozen search budget")

    def predict(
        self, features: AdvantageFeatures,
        evidence_features: AggregateEvidenceFeatures | None = None,
    ) -> tuple[float, float, float]:
        if self.evidence_benefit_head is not None:
            if evidence_features is None:
                raise ValueError("aggregate risk replay requires evidence features")
            benefit = self.evidence_benefit_head.predict(evidence_features)
            harm = self.evidence_harm_head.predict(evidence_features)
        else:
            benefit = self.benefit_head.predict(features)
            harm = self.harm_head.predict(features)
        margin = benefit - self.risk_penalty * harm - self.decision_boundary
        for region in self.regions:
            if region.matches(features) and region.margin > margin:
                benefit = region.expected_benefit
                harm = region.corruption_risk
                margin = region.margin
        return benefit, harm, margin

    def to_dict(self) -> dict[str, Any]:
        result = {
            "benefit_head": self.benefit_head.to_dict(),
            "harm_head": self.harm_head.to_dict(),
            "risk_penalty": self.risk_penalty,
            "decision_boundary": self.decision_boundary,
            "regions": [region.to_dict() for region in self.regions],
            "minimum_observations": self.minimum_observations,
            "maximum_observations": self.maximum_observations,
            "minimum_agreeing_views": self.minimum_agreeing_views,
        }
        if self.evidence_benefit_head is not None:
            result["evidence_benefit_head"] = self.evidence_benefit_head.to_dict()
            result["evidence_harm_head"] = self.evidence_harm_head.to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RiskCalibrator:
        if not isinstance(payload, Mapping):
            raise TypeError("risk calibrator payload must be a mapping")
        regions = payload.get("regions", [])
        if not isinstance(regions, list):
            raise ValueError("risk regions payload must be a list")
        benefit = payload.get("evidence_benefit_head")
        harm = payload.get("evidence_harm_head")
        return cls(
            benefit_head=RiskLinearHead.from_dict(payload.get("benefit_head")),
            harm_head=RiskLinearHead.from_dict(payload.get("harm_head")),
            risk_penalty=payload.get("risk_penalty"),
            decision_boundary=payload.get("decision_boundary"),
            regions=tuple(RiskRegion.from_dict(value) for value in regions),
            evidence_benefit_head=(
                None if benefit is None else RiskLogisticHead.from_dict(benefit)
            ),
            evidence_harm_head=(
                None if harm is None else RiskLogisticHead.from_dict(harm)
            ),
            minimum_observations=payload.get("minimum_observations", 1),
            maximum_observations=payload.get("maximum_observations", 14),
            minimum_agreeing_views=payload.get("minimum_agreeing_views", 1),
        )


@dataclass(frozen=True)
class UnifiedPolicy:
    """The answer-free configuration frozen before locked replay."""

    profile: str
    threshold: float
    raw_support_floor: float
    utility_calibrator: UtilityIsotonicCalibrator
    payload_sha256: str | None = None
    legacy_hard_gates: bool = False
    risk_calibrators: tuple[tuple[str, RiskCalibrator], ...] = ()

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError("unified policy profile is not declared")
        threshold = _unit_float(self.threshold, "threshold")
        if threshold not in THRESHOLDS:
            raise ValueError("unified policy threshold is outside the frozen grid")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(
            self,
            "raw_support_floor",
            _unit_float(self.raw_support_floor, "raw_support_floor"),
        )
        if not isinstance(self.utility_calibrator, UtilityIsotonicCalibrator):
            raise TypeError("utility_calibrator must be frozen isotonic utility")
        self.utility_calibrator._validate()
        if self.payload_sha256 is not None:
            object.__setattr__(
                self,
                "payload_sha256",
                _sha256(self.payload_sha256, "unified policy hash"),
            )
        if type(self.legacy_hard_gates) is not bool:
            raise TypeError("legacy_hard_gates must be an exact boolean")
        keys = []
        for key, calibrator in self.risk_calibrators:
            if not isinstance(key, str) or key.count("/") != 1:
                raise ValueError("risk stratum key must be backbone/answer_type")
            if not isinstance(calibrator, RiskCalibrator):
                raise TypeError("risk strata must contain frozen calibrators")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError("risk stratum keys must be unique")

    @property
    def weights(self) -> tuple[float, ...]:
        return PROFILES[self.profile]

    def predict_risk(
        self, features: AdvantageFeatures, backbone: str, answer_type: str,
        evidence_features: AggregateEvidenceFeatures | None = None,
    ) -> tuple[float, float, float]:
        if not isinstance(backbone, str) or not backbone:
            raise ValueError("risk backbone must be nonempty")
        if not isinstance(answer_type, str) or not answer_type:
            raise ValueError("risk answer type must be nonempty")
        return self.risk_calibrator_for(backbone, answer_type).predict(
            features, evidence_features,
        )

    def risk_calibrator_for(
        self, backbone: str, answer_type: str,
    ) -> RiskCalibrator:
        if not isinstance(backbone, str) or not backbone:
            raise ValueError("risk backbone must be nonempty")
        if not isinstance(answer_type, str) or not answer_type:
            raise ValueError("risk answer type must be nonempty")
        lookup = dict(self.risk_calibrators)
        for key in (
            f"{backbone}/{answer_type}", f"{backbone}/*", "*/*",
        ):
            if key in lookup:
                return lookup[key]
        raise ValueError("no hierarchical risk calibrator matches the row")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UnifiedPolicy:
        if not isinstance(payload, Mapping):
            raise TypeError("unified policy payload must be a mapping")
        risk_payload = payload.get("risk_calibrators", {})
        if not isinstance(risk_payload, Mapping):
            raise ValueError("risk calibrators payload must be a mapping")
        return cls(
            profile=payload.get("profile"),
            threshold=payload.get("threshold"),
            raw_support_floor=payload.get("raw_support_floor"),
            utility_calibrator=UtilityIsotonicCalibrator.from_dict(
                payload.get("utility_calibrator"),
            ),
            payload_sha256=payload.get("payload_sha256"),
            legacy_hard_gates=payload.get("schema_version", 2) < 3,
            risk_calibrators=tuple(
                (key, RiskCalibrator.from_dict(value))
                for key, value in sorted(risk_payload.items())
            ),
        )


@dataclass(frozen=True)
class CandidateSnapshot:
    """One changed candidate's independent evidence at a checkpoint."""

    branch_index: int
    revealed_roles: tuple[str, ...]
    observations: int
    output: Any
    canonical_answer: Any
    agreeing_hashes: tuple[str, ...]
    parseable_count: int
    raw_support_floor: float
    features: AdvantageFeatures
    evidence_features: AggregateEvidenceFeatures
    raw_score: float
    calibrated_advantage: float
    structurally_eligible: bool
    uses_aggregate_risk: bool = False
    expected_benefit: float | None = None
    corruption_risk: float | None = None


@dataclass(frozen=True)
class _View:
    branch_index: int
    role: str
    render_sha256: str
    raw_support: float
    calibrated_support: float
    output: Any
    canonical_answer: Any


@dataclass(frozen=True)
class _Branch:
    visit_index: int
    observed_path: tuple[int, ...]
    roles: tuple[str, ...]
    views: tuple[_View, ...]


@dataclass(frozen=True)
class _PreparedReplay:
    stage2: Mapping[str, Any]
    phase1_digest: str
    p0_support: float
    p0_uncertainty: float
    p0_canonical: Any
    answer_type: str
    branches: tuple[_Branch, ...]


def _parse_branches(
    row: Mapping[str, Any],
    audit: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration,
) -> tuple[_Branch, ...]:
    roots = audit.get("root_ranked_siblings")
    branches = audit.get("branches")
    if not isinstance(roots, list) or len(roots) != 4:
        raise ValueError("unified replay requires four frozen ranked roots")
    root_paths = []
    for root in roots:
        path = root.get("path") if isinstance(root, Mapping) else None
        if (
            not isinstance(path, list)
            or any(type(value) is not int or value < 0 for value in path)
        ):
            raise ValueError("frozen root path is invalid")
        root_paths.append(path)
    if not isinstance(branches, list) or len(branches) != 6:
        raise ValueError("unified replay requires the exact six frozen branches")

    seen_hashes: set[str] = set()
    parsed = []
    for index, branch in enumerate(branches):
        if (
            not isinstance(branch, Mapping)
            or branch.get("visit_index") != index
            or type(branch.get("backtracked")) is not bool
        ):
            raise ValueError("frozen branch visit provenance is invalid")
        path = branch.get("observed_path")
        if (
            not isinstance(path, list)
            or any(type(value) is not int or value < 0 for value in path)
        ):
            raise ValueError("frozen branch path is invalid")
        roles = (
            ("tight", "context")
            if index < 4 else ("tight", "medium", "context")
        )
        if index >= 4 and (
            branch["backtracked"] is not True
            or path[:1] != root_paths[index - 2]
        ):
            raise ValueError("frozen rescue-root provenance is invalid")
        views = []
        for role in roles:
            view = branch.get(f"{role}_view")
            if (
                not isinstance(view, Mapping)
                or view.get("role") != role
                or view.get("patch_path") != path
            ):
                raise ValueError("frozen view provenance is invalid")
            digest = _sha256(view.get("render_sha256"), "frozen render hash")
            if digest in seen_hashes:
                raise ValueError("frozen render hashes must be globally distinct")
            seen_hashes.add(digest)
            raw_support = _unit(view.get("raw_support"), "frozen raw support")
            calibrated_support = _unit(
                calibration.predict(raw_support), "frozen calibrated support",
            )
            try:
                record = _answer_record(row, view.get("answer"))
                output = copy.deepcopy(record.output)
                canonical_answer = (
                    None
                    if record.aggregation_available is False
                    else copy.deepcopy(record.canonical_answer)
                )
            except (TypeError, ValueError):
                output = copy.deepcopy(view.get("answer"))
                canonical_answer = None
            views.append(_View(
                branch_index=index,
                role=role,
                render_sha256=digest,
                raw_support=raw_support,
                calibrated_support=calibrated_support,
                output=output,
                canonical_answer=canonical_answer,
            ))
        parsed.append(_Branch(
            visit_index=index,
            observed_path=tuple(path),
            roles=roles,
            views=tuple(views),
        ))
    return tuple(parsed)


def _prepare_replay(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration,
) -> _PreparedReplay:
    stage2_row = sanitize_replay_row(stage2_row)
    split_row = sanitize_replay_row(split_row)
    if not isinstance(calibration, (FrozenCalibration, FrozenSelectedCalibration)):
        raise TypeError("unified replay calibration must be frozen")
    phase1_digest = _rank_digest(stage2_row)
    if phase1_digest is None or phase1_digest != _rank_digest(split_row):
        raise ValueError("Phase-1 ranking drifted")
    stage2 = replay_adaptive_search(stage2_row, split_row, calibration)
    audit = _split_audit(split_row)
    if not isinstance(audit, Mapping):
        raise ValueError("SPLIT audit is unavailable")
    if _sha256(audit.get("rank_sha256"), "split rank hash") != phase1_digest:
        raise ValueError("SPLIT rank binding drifted")
    _sha256(audit.get("query_sha256"), "split query hash")
    no_op_reason = audit.get("no_op_reason")
    if isinstance(no_op_reason, str) and no_op_reason:
        raise ValueError(f"frozen SPLIT no-op: {no_op_reason}")
    branches = _parse_branches(split_row, audit, calibration)
    p0_support = _unit(
        _stage2_support(stage2, audit, calibration), "P0 calibrated support",
    )
    p0_stability = audit.get("p0_stability")
    if not isinstance(p0_stability, Mapping):
        raise ValueError("SPLIT audit lacks P0 stability")
    p0_uncertainty = _unit(
        p0_stability.get("uncertainty"), "P0 uncertainty",
    )
    projection = dict(split_row)
    projection["output"] = copy.deepcopy(stage2["selected_output"])
    p0_canonical = _p0_canonical_answer(projection)
    if p0_canonical is None:
        raise ValueError("Stage-2 output is not canonically parseable")
    return _PreparedReplay(
        stage2=stage2,
        phase1_digest=phase1_digest,
        p0_support=p0_support,
        p0_uncertainty=p0_uncertainty,
        p0_canonical=copy.deepcopy(p0_canonical),
        answer_type=split_row["answer_type"],
        branches=branches,
    )


def _snapshots(
    prepared: _PreparedReplay,
    branch: _Branch,
    revealed_count: int,
    observed_views: Sequence[_View],
    policy: UnifiedPolicy,
    backbone: str | None,
) -> tuple[CandidateSnapshot, ...]:
    parseable = [
        view for view in observed_views if view.canonical_answer is not None
    ]
    groups: dict[str, list[_View]] = {}
    for view in parseable:
        if view.canonical_answer == prepared.p0_canonical:
            continue
        groups.setdefault(_canonical_json(view.canonical_answer), []).append(view)
    if not groups:
        return ()
    p0_observed = [
        view.calibrated_support
        for view in parseable
        if view.canonical_answer == prepared.p0_canonical
    ]
    p0_conflict = max(p0_observed, default=prepared.p0_support)
    p0_count = sum(
        view.canonical_answer == prepared.p0_canonical for view in parseable
    )
    result = []
    for key in sorted(groups):
        agreeing = groups[key]
        strongest_competitor = max(
            (
                p0_count,
                *(len(values) for other, values in groups.items() if other != key),
            ),
            default=0,
        )
        support = min(view.calibrated_support for view in agreeing)
        features = AdvantageFeatures(
            uncertainty=prepared.p0_uncertainty,
            agreement=(
                len(agreeing) / len(parseable)
                if policy.legacy_hard_gates else
                len(agreeing) / max(
                    2, len(agreeing) + strongest_competitor,
                )
            ),
            support=support,
            support_gain_01=(support - prepared.p0_support + 1.0) / 2.0,
            conflict_margin_01=(support - p0_conflict + 1.0) / 2.0,
        )
        mean_support = math.fsum(
            view.calibrated_support for view in agreeing
        ) / len(agreeing)
        dispersion = math.sqrt(math.fsum(
            (view.calibrated_support - mean_support) ** 2
            for view in agreeing
        ) / len(agreeing))
        recent = [
            view for view in observed_views[-3:]
            if view.canonical_answer is not None
        ]
        evidence_features = AggregateEvidenceFeatures(
            uncertainty=features.uncertainty,
            agreement=features.agreement,
            minimum_support=features.support,
            support_gain_01=features.support_gain_01,
            conflict_margin_01=features.conflict_margin_01,
            mean_support=mean_support,
            maximum_support=max(
                view.calibrated_support for view in agreeing
            ),
            support_dispersion=dispersion,
            agreeing_fraction=len(agreeing) / 14.0,
            parseable_fraction=len(parseable) / 14.0,
            agreeing_branch_fraction=len({
                view.branch_index for view in agreeing
            }) / 6.0,
            agreeing_role_fraction=len({
                view.role for view in agreeing
            }) / 3.0,
            recent_agreement=sum(
                view.canonical_answer == agreeing[0].canonical_answer
                for view in recent
            ) / max(1, len(recent)),
            observation_fraction=len(observed_views) / 14.0,
        )
        score = raw_advantage(features, policy.weights)
        risk_calibrator = None
        if policy.risk_calibrators:
            if backbone is None:
                raise ValueError("risk replay requires a backbone identity")
            risk_calibrator = policy.risk_calibrator_for(
                backbone, prepared.answer_type,
            )
            expected_benefit, corruption_risk, calibrated_advantage = (
                policy.predict_risk(
                    features, backbone, prepared.answer_type,
                    evidence_features,
                )
            )
        else:
            expected_benefit = corruption_risk = None
            calibrated_advantage = (
                2.0 * policy.utility_calibrator.predict(score) - 1.0
            )
        distinct_hashes = {
            view.render_sha256 for view in agreeing
        }
        minimum_raw_support = min(view.raw_support for view in agreeing)
        result.append(CandidateSnapshot(
            branch_index=branch.visit_index,
            revealed_roles=branch.roles[:revealed_count],
            observations=len(observed_views),
            output=copy.deepcopy(agreeing[0].output),
            canonical_answer=copy.deepcopy(agreeing[0].canonical_answer),
            agreeing_hashes=tuple(view.render_sha256 for view in agreeing),
            parseable_count=len(parseable),
            raw_support_floor=minimum_raw_support,
            features=features,
            evidence_features=evidence_features,
            raw_score=score,
            calibrated_advantage=calibrated_advantage,
            structurally_eligible=(
                len(distinct_hashes) >= 2
                and minimum_raw_support >= policy.raw_support_floor
                if policy.legacy_hard_gates else
                (
                    len(distinct_hashes)
                    >= risk_calibrator.minimum_agreeing_views
                    and len(observed_views)
                    >= risk_calibrator.minimum_observations
                )
                if risk_calibrator is not None else
                len(distinct_hashes) >= 1
            ),
            uses_aggregate_risk=(
                risk_calibrator is not None
                and risk_calibrator.evidence_benefit_head is not None
            ),
            expected_benefit=expected_benefit,
            corruption_risk=corruption_risk,
        ))
    if policy.legacy_hard_gates:
        return (min(
            result,
            key=lambda snapshot: (
                -len(snapshot.agreeing_hashes),
                -snapshot.features.support,
                _canonical_json(snapshot.canonical_answer),
            ),
        ),)
    return tuple(result)


def _best_snapshot(
    snapshots: Sequence[CandidateSnapshot],
) -> CandidateSnapshot | None:
    if not snapshots:
        return None
    return min(
        snapshots,
        key=lambda snapshot: (
            -snapshot.calibrated_advantage,
            -snapshot.raw_score,
            -len(snapshot.agreeing_hashes),
            _canonical_json(snapshot.canonical_answer),
        ),
    )


def candidate_snapshots(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration,
    policy: UnifiedPolicy,
    *,
    backbone: str | None = None,
) -> tuple[CandidateSnapshot, ...]:
    """Return every changed candidate ledger without evaluator labels."""
    if not isinstance(policy, UnifiedPolicy):
        raise TypeError("unified policy must be frozen")
    prepared = _prepare_replay(stage2_row, split_row, calibration)
    result = []
    observed_views = []
    for branch in prepared.branches:
        for revealed_count in range(1, len(branch.views) + 1):
            view = branch.views[revealed_count - 1]
            observed_views.append(view)
            if view.canonical_answer is None:
                break
            snapshots = _snapshots(
                prepared, branch, revealed_count, observed_views, policy,
                backbone,
            )
            result.extend(snapshots)
    return tuple(result)


def _transition(
    *, state: str, branch: int | None, revealed_roles: Sequence[str],
    snapshot: CandidateSnapshot | None, action: str, reason: str,
) -> dict[str, Any]:
    result = {
        "state": state,
        "branch": branch,
        "revealed_roles": list(revealed_roles),
        "canonical_candidate": (
            None if snapshot is None
            else copy.deepcopy(snapshot.canonical_answer)
        ),
        "features": (
            None if snapshot is None else asdict(snapshot.features)
        ),
        "raw_score": None if snapshot is None else snapshot.raw_score,
        "calibrated_advantage": (
            None if snapshot is None else snapshot.calibrated_advantage
        ),
        "action": action,
        "reason": reason,
    }
    if snapshot is not None and snapshot.expected_benefit is not None:
        result["expected_benefit"] = snapshot.expected_benefit
        result["corruption_risk"] = snapshot.corruption_risk
    if snapshot is not None and snapshot.uses_aggregate_risk:
        result["evidence_features"] = asdict(snapshot.evidence_features)
    return result


def _result(
    prepared: _PreparedReplay | None,
    stage2_output: Any,
    policy: UnifiedPolicy,
    *, selected_output: Any, selected_source: str, reason: str,
    selected_branch: int | None, observations: int,
    transitions: list[dict[str, Any]], calibration_sha256: str | None,
    failure_detail: str | None = None,
    fallback_stage2_source: str = "P0",
) -> dict[str, Any]:
    result = {
        "selected_output": copy.deepcopy(selected_output),
        "selected_source": selected_source,
        "reason": reason,
        "stage2_selected_output": copy.deepcopy(stage2_output),
        "stage2_selected_source": (
            fallback_stage2_source
            if prepared is None else prepared.stage2.get("selected_source", "P0")
        ),
        "phase1_rank_digest": (
            None if prepared is None else prepared.phase1_digest
        ),
        "calibration_manifest_sha256": calibration_sha256,
        "policy_sha256": policy.payload_sha256,
        "profile": policy.profile,
        "threshold": policy.threshold,
        "raw_support_floor": policy.raw_support_floor,
        "selected_branch": selected_branch,
        "used_backtrack": selected_branch is not None and selected_branch > 0,
        "observations": observations,
        "transitions": transitions,
    }
    if failure_detail is not None:
        result["failure_detail"] = failure_detail
    _canonical_json(result)
    return result


def fail_closed_uncertainty_support(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration | None,
    policy: UnifiedPolicy,
    detail: str,
) -> dict[str, Any]:
    """Return the exact reconstructed Stage-2 output after provenance failure."""
    if not isinstance(policy, UnifiedPolicy):
        raise TypeError("unified policy must be frozen")
    if not isinstance(detail, str) or not detail:
        raise ValueError("fail-closed detail must be nonempty")
    fallback_stage2 = {
        "selected_output": (
            copy.deepcopy(stage2_row.get("output"))
            if isinstance(stage2_row, Mapping) else None
        ),
        "selected_source": "P0",
    }
    if isinstance(calibration, (FrozenCalibration, FrozenSelectedCalibration)):
        try:
            fallback_stage2 = replay_adaptive_search(
                sanitize_replay_row(stage2_row),
                sanitize_replay_row(stage2_row),
                calibration,
            )
        except (KeyError, TypeError, ValueError):
            pass
    transitions = [_transition(
        state="P0", branch=None, revealed_roles=(), snapshot=None,
        action="FALLBACK_P0", reason=f"invalid_frozen_inputs: {detail}",
    )]
    return _result(
        None, fallback_stage2["selected_output"], policy,
        selected_output=fallback_stage2["selected_output"],
        selected_source="P0", reason="invalid_frozen_inputs",
        selected_branch=None, observations=0, transitions=transitions,
        calibration_sha256=(
            getattr(calibration, "manifest_sha256", None)
            if calibration is not None else None
        ),
        failure_detail=detail,
        fallback_stage2_source=fallback_stage2.get("selected_source", "P0"),
    )


def replay_uncertainty_support(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration | None,
    policy: UnifiedPolicy,
    *,
    backbone: str | None = None,
) -> dict[str, Any]:
    """Replay STOP/CONTINUE/BACKTRACK/REPLACE using one utility score."""
    if not isinstance(policy, UnifiedPolicy):
        raise TypeError("unified policy must be frozen")
    try:
        if calibration is None:
            raise ValueError("support calibration is unavailable")
        prepared = _prepare_replay(stage2_row, split_row, calibration)
    except (KeyError, TypeError, ValueError) as error:
        return fail_closed_uncertainty_support(
            stage2_row, split_row, calibration, policy, str(error),
        )

    stage2_output = prepared.stage2["selected_output"]
    transitions = [_transition(
        state="P0", branch=None, revealed_roles=(), snapshot=None,
        action="OBSERVE", reason="begin_frozen_observations",
    )]
    observations = 0
    optimistic = AdvantageFeatures(
        prepared.p0_uncertainty, 1.0, 1.0, 1.0, 1.0,
    )
    if policy.risk_calibrators:
        if backbone is None:
            return fail_closed_uncertainty_support(
                stage2_row, split_row, calibration, policy,
                "risk replay requires a backbone identity",
            )
        risk_calibrator = policy.risk_calibrator_for(
            backbone, prepared.answer_type,
        )
        optimistic_advantage = max((
            1.0 - risk_calibrator.decision_boundary,
            *(region.margin for region in risk_calibrator.regions),
        ))
    else:
        optimistic_advantage = 2.0 * policy.utility_calibrator.predict(
            raw_advantage(optimistic, policy.weights),
        ) - 1.0

    observed_views = []
    for branch in prepared.branches:
        for revealed_count, view in enumerate(branch.views, start=1):
            observations += 1
            observed_views.append(view)
            snapshots = _snapshots(
                prepared, branch, revealed_count, observed_views, policy,
                backbone,
            )
            snapshot = _best_snapshot(snapshots)
            if view.canonical_answer is None:
                transitions.append(_transition(
                    state="OBSERVE", branch=branch.visit_index,
                    revealed_roles=branch.roles[:revealed_count],
                    snapshot=snapshot, action="BACKTRACK",
                    reason="unparseable_observation",
                ))
                if (
                    policy.risk_calibrators
                    and observations >= risk_calibrator.maximum_observations
                ):
                    transitions.append(_transition(
                        state="P0", branch=None, revealed_roles=(),
                        snapshot=None, action="STOP_P0",
                        reason="observation_budget_exhausted",
                    ))
                    return _result(
                        prepared, stage2_output, policy,
                        selected_output=stage2_output,
                        selected_source="P0",
                        reason="observation_budget_exhausted",
                        selected_branch=None, observations=observations,
                        transitions=transitions,
                        calibration_sha256=calibration.manifest_sha256,
                    )
                break
            if (
                snapshot is not None
                and snapshot.structurally_eligible
                and snapshot.calibrated_advantage >= policy.threshold
            ):
                transitions.append(_transition(
                    state="OBSERVE", branch=branch.visit_index,
                    revealed_roles=branch.roles[:revealed_count],
                    snapshot=snapshot, action="REPLACE",
                    reason="calibrated_utility_reached",
                ))
                output = _candidate_output(
                    split_row, stage2_output, prepared.p0_canonical,
                    {"tight": {"output": snapshot.output}},
                    snapshot.canonical_answer,
                )
                return _result(
                    prepared, stage2_output, policy,
                    selected_output=output, selected_source="SPLIT",
                    reason="calibrated_utility_reached",
                    selected_branch=branch.visit_index,
                    observations=observations, transitions=transitions,
                    calibration_sha256=calibration.manifest_sha256,
                )
            if (
                policy.risk_calibrators
                and observations >= risk_calibrator.maximum_observations
            ):
                transitions.append(_transition(
                    state="P0", branch=None, revealed_roles=(),
                    snapshot=None, action="STOP_P0",
                    reason="observation_budget_exhausted",
                ))
                return _result(
                    prepared, stage2_output, policy,
                    selected_output=stage2_output, selected_source="P0",
                    reason="observation_budget_exhausted",
                    selected_branch=None, observations=observations,
                    transitions=transitions,
                    calibration_sha256=calibration.manifest_sha256,
                )
            if revealed_count < len(branch.views) and (
                optimistic_advantage >= policy.threshold
            ):
                transitions.append(_transition(
                    state="OBSERVE", branch=branch.visit_index,
                    revealed_roles=branch.roles[:revealed_count],
                    snapshot=snapshot, action="CONTINUE",
                    reason="utility_upper_bound_reachable",
                ))
                continue
            transitions.append(_transition(
                state="OBSERVE", branch=branch.visit_index,
                revealed_roles=branch.roles[:revealed_count],
                snapshot=snapshot, action="BACKTRACK",
                reason=(
                    "branch_exhausted"
                    if revealed_count == len(branch.views)
                    else "utility_upper_bound_below_threshold"
                ),
            ))
            break

    transitions.append(_transition(
        state="P0", branch=None, revealed_roles=(), snapshot=None,
        action="STOP_P0", reason="all_branches_exhausted",
    ))
    return _result(
        prepared, stage2_output, policy,
        selected_output=stage2_output, selected_source="P0",
        reason="all_branches_exhausted", selected_branch=None,
        observations=observations, transitions=transitions,
        calibration_sha256=calibration.manifest_sha256,
    )
