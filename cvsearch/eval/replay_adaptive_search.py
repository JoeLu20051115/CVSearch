"""Label-blind replay selection over candidate-only scale observations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from cvsearch.evidence_gap.adaptive_controller import (
    IsotonicCalibrator,
    SupportObservation,
    fit_isotonic,
    infer_evidence_demand,
    select_adaptive_action,
)
from cvsearch.evidence_gap.answers import aggregate_hr_answers, aggregate_vstar_losses
from cvsearch.evidence_gap.types import EXPAND, ZOOM


_ACTIONS = (ZOOM, EXPAND)
_CALIBRATION_KEYS = frozenset({"row_id", "raw_support", "support_sufficient"})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


@dataclass(frozen=True)
class FrozenCalibration:
    calibrator: IsotonicCalibrator
    sample_count: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.calibrator, IsotonicCalibrator):
            raise TypeError("calibrator must be isotonic")
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive exact integer")
        if type(self.manifest_sha256) is not str or len(self.manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must be a SHA-256 hex digest")
        int(self.manifest_sha256, 16)
        if self.manifest_sha256 != _sha256(self._payload()):
            raise ValueError("calibration manifest hash does not match payload")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sample_count": self.sample_count,
            "calibrator": self.calibrator.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload(), manifest_sha256=self.manifest_sha256)


def freeze_isotonic_calibration(
    rows: Sequence[Mapping[str, Any]],
) -> FrozenCalibration:
    """Fit support calibration from an exact schema with no answer labels."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("calibration rows must be a sequence")
    seen: set[str] = set()
    samples: list[tuple[float, int]] = []
    for row in rows:
        if type(row) is not dict or set(row) != _CALIBRATION_KEYS:
            raise ValueError("calibration rows must use the exact support-only schema")
        row_id = row["row_id"]
        if type(row_id) is not str or not row_id or row_id in seen:
            raise ValueError("calibration row_id must be unique and nonempty")
        seen.add(row_id)
        score = _unit(row["raw_support"], "raw_support")
        label = row["support_sufficient"]
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("support_sufficient must be an exact binary integer")
        samples.append((score, label))
    calibrator = fit_isotonic(tuple(samples))
    payload = {
        "schema_version": 1,
        "sample_count": len(samples),
        "calibrator": calibrator.to_dict(),
    }
    return FrozenCalibration(calibrator, len(samples), _sha256(payload))


@dataclass(frozen=True)
class ReplayCandidate:
    action: str
    output: Any
    canonical_answer: Any
    current_observation: SupportObservation
    candidate_observation: SupportObservation

    def to_dict(
        self, calibrator: IsotonicCalibrator, prior: Mapping[str, float],
    ) -> dict[str, Any]:
        current = calibrator.predict(self.current_observation.raw_support)
        candidate = calibrator.predict(self.candidate_observation.raw_support)
        gain = candidate - current
        score = (
            candidate
            + 0.05 * self.candidate_observation.answer_consistency
            + 0.05 * prior.get(self.action, 0.0)
            - 0.05 * self.candidate_observation.normalized_cost
        )
        return {
            "action": self.action,
            "output": copy.deepcopy(self.output),
            "canonical_answer": copy.deepcopy(self.canonical_answer),
            "raw_current_support": self.current_observation.raw_support,
            "raw_candidate_support": self.candidate_observation.raw_support,
            "calibrated_current_support": current,
            "calibrated_candidate_support": candidate,
            "calibrated_support_gain": gain,
            "answer_consistency": self.candidate_observation.answer_consistency,
            "normalized_cost": self.candidate_observation.normalized_cost,
            "selection_score": score,
        }


def _rank_digest(row: Mapping[str, Any]) -> str | None:
    trace = row.get("method_trace")
    if type(trace) is not dict or type(trace.get("candidate_ranks")) is not list:
        return None
    return _sha256(trace["candidate_ranks"])


def _same_json(left: Any, right: Any) -> bool:
    try:
        return _canonical_json(left) == _canonical_json(right)
    except (TypeError, ValueError):
        return False


def _answer_record(row: Mapping[str, Any], candidate_answer: Any):
    answer_type = row.get("answer_type")
    if answer_type == "option_list":
        options = row.get("options")
        if type(options) is not list or type(candidate_answer) is not list:
            raise ValueError("HR candidate requires option blocks and raw outputs")
        return aggregate_hr_answers(list(options), list(candidate_answer))
    if answer_type == "logits_match":
        if type(candidate_answer) is not dict or type(candidate_answer.get("losses")) is not list:
            raise ValueError("V* candidate requires a loss vector")
        record = aggregate_vstar_losses([candidate_answer["losses"]])
        if candidate_answer.get("winner") != record.output:
            raise ValueError("V* candidate winner does not match losses")
        return record
    raise ValueError("unsupported answer_type")


def _extract_candidates(row: Mapping[str, Any]) -> tuple[ReplayCandidate, ...]:
    trace = row.get("method_trace")
    if type(trace) is not dict or type(trace.get("steps")) is not list:
        return ()
    candidates: list[ReplayCandidate] = []
    for step in trace["steps"]:
        if type(step) is not dict or step.get("action") not in _ACTIONS:
            continue
        action = step["action"]
        audit = step.get("zoom_audit" if action == ZOOM else "expand_audit")
        if type(audit) is not dict:
            continue
        batch = audit.get("batch_result")
        if type(batch) is not dict or batch.get("status") != "success":
            continue
        current = audit.get("current_gap_support")
        candidate = audit.get("candidate_gap_support")
        stability = audit.get("candidate_stability")
        p0_stability = audit.get("p0_stability")
        if not all(type(value) is dict for value in (
            current, candidate, stability, p0_stability,
        )):
            continue
        try:
            current_p = _unit(current["p_yes"], "current p_yes")
            candidate_p = _unit(candidate["p_yes"], "candidate p_yes")
            current_frequency = _unit(
                p0_stability["frequency"], "current answer consistency",
            )
            cost = _unit(audit["normalized_actual_cost"], "normalized cost")
            record = _answer_record(row, batch.get("candidate_answer"))
            answer_consistency = _unit(record.frequency, "candidate answer consistency")
        except (KeyError, TypeError, ValueError):
            continue
        if record.aggregation_available is False:
            continue
        candidates.append(ReplayCandidate(
            action=action,
            output=copy.deepcopy(record.output),
            canonical_answer=copy.deepcopy(record.canonical_answer),
            current_observation=SupportObservation(
                p_full=current_p, p_partial=0.0, p_none=1.0 - current_p,
                support_consistency=1.0,
                answer_consistency=current_frequency,
                missing_reason="none", normalized_cost=0.0,
            ),
            candidate_observation=SupportObservation(
                p_full=candidate_p, p_partial=0.0, p_none=1.0 - candidate_p,
                support_consistency=1.0,
                answer_consistency=answer_consistency,
                missing_reason="none", normalized_cost=cost,
            ),
        ))
    return tuple(candidates)


def _fallback(
    phase1_row: Mapping[str, Any], reason: str, *, rank_digest: str | None,
    demand: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "selected_output": copy.deepcopy(phase1_row.get("output")),
        "selected_source": "P0",
        "reason": reason,
        "phase1_rank_digest": rank_digest,
        "demand": None if demand is None else dict(demand),
        "controller_decision": None,
        "candidates": [],
    }


def replay_adaptive_search(
    phase1_row: Mapping[str, Any],
    observation_row: Mapping[str, Any],
    calibration: FrozenCalibration | None,
    *,
    minimum_gain: float = 0.05,
    minimum_support: float = 0.5,
    minimum_answer_consistency: float = 0.75,
) -> dict[str, Any]:
    """Select an observed action without reading benchmark or answer truth."""
    if type(phase1_row) is not dict or type(observation_row) is not dict:
        raise TypeError("replay rows must be exact dictionaries")
    phase1_digest = _rank_digest(phase1_row)
    if phase1_digest is None or phase1_digest != _rank_digest(observation_row):
        return _fallback(phase1_row, "phase1_rank_drift", rank_digest=phase1_digest)
    if not _same_json(phase1_row.get("output"), observation_row.get("output")):
        return _fallback(
            phase1_row, "phase1_output_drift", rank_digest=phase1_digest,
        )
    trace = observation_row.get("method_trace")
    query_plan = trace.get("query_plan") if type(trace) is dict else None
    evidence_items = query_plan.get("evidence_items") if type(query_plan) is dict else None
    if type(evidence_items) is not list:
        return _fallback(
            phase1_row, "evidence_demand_unavailable", rank_digest=phase1_digest,
        )
    demand = infer_evidence_demand(tuple(evidence_items))
    demand_dict = demand.to_dict()
    if calibration is None:
        return _fallback(
            phase1_row, "calibration_unavailable", rank_digest=phase1_digest,
            demand=demand_dict,
        )
    if not isinstance(calibration, FrozenCalibration):
        raise TypeError("calibration must be frozen before replay")
    minimum_gain = _unit(minimum_gain, "minimum_gain")
    minimum_support = _unit(minimum_support, "minimum_support")
    minimum_answer_consistency = _unit(
        minimum_answer_consistency, "minimum_answer_consistency",
    )
    extracted = _extract_candidates(observation_row)
    if not extracted:
        return _fallback(
            phase1_row, "no_admitted_candidate", rank_digest=phase1_digest,
            demand=demand_dict,
        )

    prior = demand.action_prior()
    candidate_rows = [
        candidate.to_dict(calibration.calibrator, prior) for candidate in extracted
    ]
    first = extracted[0]
    costs = {
        candidate.action: candidate.candidate_observation.normalized_cost
        for candidate in extracted
    }
    controller = select_adaptive_action(
        demand,
        first.current_observation,
        available=tuple(dict.fromkeys(candidate.action for candidate in extracted)),
        trajectory=(first.current_observation.raw_support,),
        has_unvisited_branch=False,
        calibrated_support=calibration.calibrator.predict(
            first.current_observation.raw_support,
        ),
        action_costs=costs,
    )
    eligible = [
        row for row in candidate_rows
        if row["calibrated_support_gain"] >= minimum_gain
        and row["calibrated_candidate_support"] >= minimum_support
        and row["answer_consistency"] >= minimum_answer_consistency
    ]
    if not eligible:
        result = _fallback(
            phase1_row, "insufficient_calibrated_gain", rank_digest=phase1_digest,
            demand=demand_dict,
        )
        result["controller_decision"] = controller.to_dict()
        result["candidates"] = candidate_rows
        return result
    best = min(
        eligible,
        key=lambda row: (
            -row["selection_score"], -row["calibrated_support_gain"],
            _ACTIONS.index(row["action"]),
        ),
    )
    result = {
        "selected_output": copy.deepcopy(best["output"]),
        "selected_source": best["action"],
        "reason": "calibrated_support_gain",
        "phase1_rank_digest": phase1_digest,
        "demand": demand_dict,
        "controller_decision": controller.to_dict(),
        "candidates": candidate_rows,
    }
    _canonical_json(result)
    return result


__all__ = [
    "FrozenCalibration", "ReplayCandidate", "freeze_isotonic_calibration",
    "replay_adaptive_search",
]
