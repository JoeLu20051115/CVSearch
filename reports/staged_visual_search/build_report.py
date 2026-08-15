#!/usr/bin/env python3
"""Build the staged adaptive visual-search method report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


EVIDENCE_PATHS = {
    "stage1_baseline": "reproduction/evidence_gap/reports/ranking-recall-baseline.json",
    "stage1_v1": "reproduction/evidence_gap/reports/ranking-recall-v1.json",
    "stage1_v6": "reproduction/evidence_gap/reports/phase1-v6-cross-backbone.json",
    "stage2": "reproduction/evidence_gap/reports/adaptive-search-v7-cross-backbone.json",
    "accepted_policy": (
        "reproduction/evidence_gap/adaptive_search_v18/"
        "robust-transfer-candidate-free-policy.json"
    ),
    "mme": "reproduction/evidence_gap/reports/robust-transfer-mme-realworld-lite.json",
}


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Evidence must be a JSON object: {path}")
    return value


def _field(value: Mapping[str, Any], path: str, expected: type) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"Missing evidence field: {path}")
        current = current[part]
    if expected is float:
        if type(current) not in (int, float):
            raise ValueError(f"Evidence field is not numeric: {path}")
        return float(current)
    if type(current) is not expected:
        raise ValueError(f"Evidence field has wrong type: {path}")
    return current


def collect_evidence(repo_root: Path) -> dict[str, object]:
    """Read the frozen experiment reports used by pages 14 and 15."""

    reports = {
        name: _load_object(repo_root / relative_path)
        for name, relative_path in EVIDENCE_PATHS.items()
    }

    baseline = reports["stage1_baseline"]
    selected = reports["stage1_v1"]
    v6 = reports["stage1_v6"]
    stage2 = reports["stage2"]
    accepted = reports["accepted_policy"]
    mme = reports["mme"]

    qwen_v6 = _field(v6, "results.qwen2_5_vl_7b", dict)
    qwen_stage2 = _field(stage2, "cells.qwen2_5_vl_7b", dict)
    accepted_metrics = _field(accepted, "accepted_outer_checkpoint.metrics", dict)
    mme_qwen = _field(mme, "backbones.qwen", dict)
    mme_internvl = _field(mme, "backbones.internvl", dict)

    evidence: dict[str, object] = {
        "stage1_dev": {
            "baseline_r3": _field(baseline, "topic.recall_at.3", float),
            "selected_r3": _field(selected, "topic.recall_at.3", float),
            "baseline_acc": _field(baseline, "answer_accuracy", float),
            "selected_acc": _field(selected, "answer_accuracy", float),
        },
        "stage1_dev_r1": (
            _field(baseline, "topic.recall_at.1", float),
            _field(selected, "topic.recall_at.1", float),
        ),
        "stage1_v6_qwen": {
            "baseline_r1": _field(qwen_v6, "original.recall_at_1", float),
            "selected_r1": _field(qwen_v6, "v6.recall_at_1", float),
            "baseline_r3": _field(qwen_v6, "original.recall_at_3", float),
            "selected_r3": _field(qwen_v6, "v6.recall_at_3", float),
            "r3_delta": _field(qwen_v6, "recall_at_3_delta", float),
            "baseline_acc": _field(qwen_v6, "original.answer_accuracy", float),
            "selected_acc": _field(qwen_v6, "v6.answer_accuracy", float),
            "acc_delta": _field(qwen_v6, "answer_accuracy_delta", float),
        },
        "stage2_qwen": {
            "vstar": (
                _field(qwen_stage2, "vstar.baseline_correct", int),
                _field(qwen_stage2, "vstar.selected_correct", int),
            ),
            "treebench": (
                _field(qwen_stage2, "treebench.baseline_correct", int),
                _field(qwen_stage2, "treebench.selected_correct", int),
            ),
            "hr_bench_4k": (
                _field(qwen_stage2, "hr_bench_4k.baseline_correct", int),
                _field(qwen_stage2, "hr_bench_4k.selected_correct", int),
            ),
            "hr_bench_8k": (
                _field(qwen_stage2, "hr_bench_8k.baseline_correct", int),
                _field(qwen_stage2, "hr_bench_8k.selected_correct", int),
            ),
        },
        "stage2_aggregate": {
            "corrections": _field(stage2, "aggregate.corrections", int),
            "corruptions": _field(stage2, "aggregate.corruptions", int),
        },
        "accepted_development_qwen_delta": _field(
            accepted_metrics, "backbone_deltas.qwen", int
        ),
        "accepted_development": {
            "net_gain": _field(accepted_metrics, "net_gain", int),
            "corrections": _field(accepted_metrics, "corrections", int),
            "corruptions": _field(accepted_metrics, "corruptions", int),
            "qwen_delta": _field(accepted_metrics, "backbone_deltas.qwen", int),
            "internvl_delta": _field(
                accepted_metrics, "backbone_deltas.internvl", int
            ),
            "topics": _field(accepted_metrics, "topics", int),
            "official_units": _field(accepted_metrics, "official_units", int),
            "mean_observations": _field(accepted_metrics, "mean_observations", float),
        },
        "mme_qwen": {
            key: _field(mme_qwen, key, float if key == "mean_observations" else int)
            for key in (
                "topics",
                "baseline_correct",
                "robust_correct",
                "delta",
                "corrections",
                "corruptions",
                "selections",
                "runtime_failures",
                "observations",
                "mean_observations",
            )
        },
        "mme_internvl": {
            key: _field(
                mme_internvl, key, float if key == "mean_observations" else int
            )
            for key in (
                "topics",
                "baseline_correct",
                "robust_correct",
                "delta",
                "corrections",
                "corruptions",
                "runtime_failures",
                "mean_observations",
            )
        },
        "mme_evaluation_count": _field(mme, "evaluation_count", int),
        "mme_gate_passed": _field(mme, "gate.passed", bool),
    }
    validate_claims(evidence)
    return evidence


def validate_claims(evidence: Mapping[str, object]) -> None:
    """Reject evidence drift that would make the report's claims inaccurate."""

    stage1_v6 = _field(evidence, "stage1_v6_qwen", dict)
    if _field(stage1_v6, "r3_delta", float) < 0:
        raise ValueError("Stage 1 v6 Qwen Recall@3 must not regress")
    if _field(stage1_v6, "acc_delta", float) < 0:
        raise ValueError("Stage 1 v6 Qwen accuracy must not regress")

    stage2 = _field(evidence, "stage2_qwen", dict)
    for unit_name, pair in stage2.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Malformed Stage 2 score pair: {unit_name}")
        if type(pair[0]) is not int or type(pair[1]) is not int:
            raise ValueError(f"Stage 2 score pair must contain integers: {unit_name}")
        if pair[1] < pair[0]:
            raise ValueError(f"Stage 2 Qwen result regressed: {unit_name}")

    accepted = _field(evidence, "accepted_development", dict)
    if _field(accepted, "qwen_delta", int) != 7:
        raise ValueError("Accepted development policy Qwen delta drifted from +7")
    if _field(accepted, "corruptions", int) != 0:
        raise ValueError("Accepted development policy must have zero corruptions")

    if _field(evidence, "mme_evaluation_count", int) != 1:
        raise ValueError("MME-RealWorld-Lite must remain a single sealed evaluation")
    if not _field(evidence, "mme_gate_passed", bool):
        raise ValueError("MME-RealWorld-Lite result did not pass its frozen gate")
    mme_qwen = _field(evidence, "mme_qwen", dict)
    if _field(mme_qwen, "delta", int) != 27:
        raise ValueError("MME Qwen end-to-end delta drifted from +27")
    if _field(mme_qwen, "runtime_failures", int) != 0:
        raise ValueError("MME Qwen evaluation contains runtime failures")
