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


@dataclass(frozen=True)
class UnifiedPolicy:
    """The answer-free configuration frozen before locked replay."""

    profile: str
    threshold: float
    raw_support_floor: float
    utility_calibrator: UtilityIsotonicCalibrator
    payload_sha256: str | None = None

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

    @property
    def weights(self) -> tuple[float, ...]:
        return PROFILES[self.profile]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UnifiedPolicy:
        if not isinstance(payload, Mapping):
            raise TypeError("unified policy payload must be a mapping")
        return cls(
            profile=payload.get("profile"),
            threshold=payload.get("threshold"),
            raw_support_floor=payload.get("raw_support_floor"),
            utility_calibrator=UtilityIsotonicCalibrator.from_dict(
                payload.get("utility_calibrator"),
            ),
            payload_sha256=payload.get("payload_sha256"),
        )


@dataclass(frozen=True)
class CandidateSnapshot:
    """One leading changed candidate at an observation checkpoint."""

    branch_index: int
    revealed_roles: tuple[str, ...]
    output: Any
    canonical_answer: Any
    agreeing_hashes: tuple[str, ...]
    parseable_count: int
    raw_support_floor: float
    features: AdvantageFeatures
    raw_score: float
    calibrated_advantage: float
    structurally_eligible: bool


@dataclass(frozen=True)
class _View:
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
        branches=branches,
    )


def _snapshot(
    prepared: _PreparedReplay,
    branch: _Branch,
    revealed_count: int,
    observed_views: Sequence[_View],
    policy: UnifiedPolicy,
) -> CandidateSnapshot | None:
    parseable = [
        view for view in observed_views if view.canonical_answer is not None
    ]
    groups: dict[str, list[_View]] = {}
    for view in parseable:
        if view.canonical_answer == prepared.p0_canonical:
            continue
        groups.setdefault(_canonical_json(view.canonical_answer), []).append(view)
    if not groups:
        return None
    agreeing = min(
        groups.values(),
        key=lambda values: (
            -len(values),
            -min(value.calibrated_support for value in values),
            _canonical_json(values[0].canonical_answer),
        ),
    )
    support = min(view.calibrated_support for view in agreeing)
    p0_observed = [
        view.calibrated_support
        for view in parseable
        if view.canonical_answer == prepared.p0_canonical
    ]
    p0_conflict = max(p0_observed, default=prepared.p0_support)
    features = AdvantageFeatures(
        uncertainty=prepared.p0_uncertainty,
        agreement=len(agreeing) / len(parseable),
        support=support,
        support_gain_01=(support - prepared.p0_support + 1.0) / 2.0,
        conflict_margin_01=(support - p0_conflict + 1.0) / 2.0,
    )
    score = raw_advantage(features, policy.weights)
    calibrated_advantage = 2.0 * policy.utility_calibrator.predict(score) - 1.0
    return CandidateSnapshot(
        branch_index=branch.visit_index,
        revealed_roles=branch.roles[:revealed_count],
        output=copy.deepcopy(agreeing[0].output),
        canonical_answer=copy.deepcopy(agreeing[0].canonical_answer),
        agreeing_hashes=tuple(view.render_sha256 for view in agreeing),
        parseable_count=len(parseable),
        raw_support_floor=min(view.raw_support for view in agreeing),
        features=features,
        raw_score=score,
        calibrated_advantage=calibrated_advantage,
        structurally_eligible=(
            len({view.render_sha256 for view in agreeing}) >= 2
            and min(view.raw_support for view in agreeing)
            >= policy.raw_support_floor
        ),
    )


def candidate_snapshots(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration,
    policy: UnifiedPolicy,
) -> tuple[CandidateSnapshot, ...]:
    """Return all leading changed candidates without consulting evaluator labels."""
    if not isinstance(policy, UnifiedPolicy):
        raise TypeError("unified policy must be frozen")
    prepared = _prepare_replay(stage2_row, split_row, calibration)
    result = []
    observed_views = []
    for branch in prepared.branches:
        for revealed_count in range(1, len(branch.views) + 1):
            observed_views.append(branch.views[revealed_count - 1])
            snapshot = _snapshot(
                prepared, branch, revealed_count, observed_views, policy,
            )
            if snapshot is not None:
                result.append(snapshot)
    return tuple(result)


def _transition(
    *, state: str, branch: int | None, revealed_roles: Sequence[str],
    snapshot: CandidateSnapshot | None, action: str, reason: str,
) -> dict[str, Any]:
    return {
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
                sanitize_replay_row(split_row),
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
    optimistic_advantage = 2.0 * policy.utility_calibrator.predict(
        raw_advantage(optimistic, policy.weights),
    ) - 1.0

    observed_views = []
    for branch in prepared.branches:
        for revealed_count, view in enumerate(branch.views, start=1):
            observations += 1
            observed_views.append(view)
            snapshot = _snapshot(
                prepared, branch, revealed_count, observed_views, policy,
            )
            if view.canonical_answer is None:
                transitions.append(_transition(
                    state="OBSERVE", branch=branch.visit_index,
                    revealed_roles=branch.roles[:revealed_count],
                    snapshot=snapshot, action="BACKTRACK",
                    reason="unparseable_observation",
                ))
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
