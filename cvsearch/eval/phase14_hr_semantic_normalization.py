"""Normalize a strict HR semantic majority back to all shuffled letters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from cvsearch.evidence_gap.answers import aggregate_hr_answers
from cvsearch.eval.phase7_uncertainty_confirmation import _p0, _snapshot_json


HR_NORMALIZATION_MIN_FREQUENCY = 0.75


@dataclass(frozen=True)
class HRSemanticNormalizationDecision:
    action: str
    status: str
    output: Any
    frequency: float | None
    canonical_answer: str | None


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "feasible": False,
        "output": None,
        "canonical_answer": None,
        "frequency": None,
        "margin": None,
        "reason": reason,
    }


def project_hr_semantic_normalization(
    option_blocks: Sequence[str], raw_outputs: Sequence[str],
) -> dict[str, Any]:
    try:
        record = aggregate_hr_answers(list(option_blocks), list(raw_outputs))
    except (TypeError, ValueError):
        return _unavailable("invalid_option_schema")
    if record.aggregation_available is not True:
        return _unavailable(record.aggregation_reason or "aggregation_unavailable")
    if record.frequency < HR_NORMALIZATION_MIN_FREQUENCY:
        return _unavailable("insufficient_semantic_majority")
    output = _snapshot_json(record.output)
    if output == list(raw_outputs):
        return _unavailable("already_canonical")
    return {
        "feasible": True,
        "output": output,
        "canonical_answer": record.canonical_answer,
        "frequency": float(record.frequency),
        "margin": float(record.margin),
        "reason": None,
    }


def select_hr_semantic_normalization(
    p0: dict[str, Any], option_blocks: Sequence[str],
) -> HRSemanticNormalizationDecision:
    p0, _ = _p0(p0)
    projection = project_hr_semantic_normalization(
        option_blocks, p0["output"],
    )
    if projection["feasible"]:
        return HRSemanticNormalizationDecision(
            "HR_NORMALIZE",
            "selected_hr_semantic_majority_normalization",
            _snapshot_json(projection["output"]),
            projection["frequency"],
            projection["canonical_answer"],
        )
    return HRSemanticNormalizationDecision(
        "P0", "retained_p0", _snapshot_json(p0["output"]), None, None,
    )
