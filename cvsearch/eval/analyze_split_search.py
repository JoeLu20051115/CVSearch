#!/usr/bin/env python3
"""Small evaluator-only helpers for Stage 3 split-search experiments."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.adaptive_controller import IsotonicCalibrator
from cvsearch.evidence_gap.answers import official_letter

from .replay_adaptive_search import FrozenSelectedCalibration, _answer_record
from .replay_split_search import select_split_candidate


def load_selected_calibrations(
    path: Path,
) -> dict[str, FrozenSelectedCalibration]:
    """Rebuild immutable per-backbone calibrators from the Phase-2 suite file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    backbones = payload.get("backbones") if isinstance(payload, Mapping) else None
    if not isinstance(backbones, Mapping) or not backbones:
        raise ValueError("calibration suite must contain nonempty backbones")
    result = {}
    for name, value in backbones.items():
        if not isinstance(name, str) or not name or not isinstance(value, Mapping):
            raise ValueError("calibration suite backbone record is invalid")
        calibrator = value.get("calibrator")
        metrics = value.get("candidate_metrics")
        if not isinstance(calibrator, Mapping) or not isinstance(metrics, list):
            raise ValueError("calibration suite record is incomplete")
        result[name] = FrozenSelectedCalibration(
            calibrator=IsotonicCalibrator(
                tuple(calibrator.get("upper_bounds", ())),
                tuple(calibrator.get("probabilities", ())),
            ),
            sample_count=value.get("sample_count"),
            source_group_count=value.get("source_group_count"),
            selected_weight=value.get("selected_weight"),
            candidate_metrics=tuple(
                (item.get("weight"), item.get("brier"), item.get("ece_10"))
                for item in metrics
            ),
            calibration_rows_sha256=value.get("calibration_rows_sha256"),
            source_groups_sha256=value.get("source_groups_sha256"),
            manifest_sha256=value.get("manifest_sha256"),
        )
    return result


def _split_audit(row: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = row.get("method_trace")
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    matches = [
        step.get("split_search_audit")
        for step in steps if isinstance(step, Mapping)
        and step.get("action") == "SPLIT"
        and isinstance(step.get("split_search_audit"), Mapping)
    ] if isinstance(steps, list) else []
    if len(matches) != 1:
        raise ValueError("row must contain exactly one split-search audit")
    return matches[0]


def candidate_outputs(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return tight/context outputs in visit order after official aggregation."""
    audit = _split_audit(row)
    branches = audit.get("branches")
    if not isinstance(branches, list) or not branches:
        return ()
    outputs = []
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ValueError("split branch is invalid")
        for role in ("tight_view", "context_view"):
            view = branch.get(role)
            if not isinstance(view, Mapping):
                raise ValueError("split view is invalid")
            record = _answer_record(row, view.get("answer"))
            outputs.append(record.output)
    return tuple(outputs)


def official_correctness(
    benchmark: str, row: Mapping[str, Any], output: Any,
) -> tuple[bool, ...]:
    """Score an output in V*, HR-Bench, or TreeBench official units."""
    if benchmark == "vstar":
        return (type(output) is int and output == 0,)
    if benchmark in {"hr_bench_4k", "hr_bench_8k"}:
        truth = row.get("answer")
        if type(truth) is not list or type(output) is not list or len(truth) != len(output):
            raise ValueError("HR truth and output must be aligned lists")
        return tuple(
            isinstance(value, str) and official_letter(value) == answer
            for value, answer in zip(output, truth)
        )
    if benchmark == "treebench":
        answer = str(row.get("answer", "")).strip().upper()
        prediction = output.strip().upper() if isinstance(output, str) else ""
        return (prediction == answer,)
    raise ValueError("unsupported Stage-3 benchmark")


def _index_rows(rows: list[Mapping[str, Any]], name: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} rows must be a nonempty list")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{name} rows must be mappings")
        ordinal = row.get("_eg_ordinal")
        if type(ordinal) is not int or ordinal < 0 or ordinal in result:
            raise ValueError(f"{name} ordinals must be unique nonnegative integers")
        result[ordinal] = row
    return result


def score_cell(
    benchmark: str,
    stage2_rows: list[Mapping[str, Any]],
    split_rows: list[Mapping[str, Any]],
    calibration: FrozenSelectedCalibration,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay and score one frozen backbone/benchmark partition."""
    baseline = _index_rows(stage2_rows, "Stage-2")
    split = _index_rows(split_rows, "split")
    if baseline.keys() != split.keys():
        raise ValueError("Stage-2 and split ordinals must align exactly")
    before_flags = []
    after_flags = []
    oracle_fixes = 0
    selections = 0
    reasons: Counter[str] = Counter()
    details = []
    for ordinal in sorted(baseline):
        row = baseline[ordinal]
        observed = split[ordinal]
        decision = select_split_candidate(row, observed, calibration, policy)
        before = official_correctness(
            benchmark, row, decision["stage2_selected_output"],
        )
        after = official_correctness(benchmark, row, decision["selected_output"])
        candidates = tuple(
            official_correctness(benchmark, row, output)
            for output in candidate_outputs(observed)
        )
        if any(len(value) != len(before) for value in candidates):
            raise AssertionError("candidate official-unit count drifted")
        oracle_fixes += sum(
            not flag and any(candidate[index] for candidate in candidates)
            for index, flag in enumerate(before)
        )
        before_flags.extend(before)
        after_flags.extend(after)
        selections += decision["selected_source"] == "SPLIT"
        reasons[decision["reason"]] += 1
        details.append({
            "ordinal": ordinal,
            "stage2_output": decision["stage2_selected_output"],
            "stage3_output": decision["selected_output"],
            "stage2_correct": list(before),
            "stage3_correct": list(after),
            "candidate_correct": [list(value) for value in candidates],
            "selected_source": decision["selected_source"],
            "reason": decision["reason"],
            "selected_branch": decision["selected_branch"],
            "used_backtrack": decision["used_backtrack"],
            "branches": decision["branches"],
        })
    stage2_correct = sum(before_flags)
    stage3_correct = sum(after_flags)
    return {
        "topics": len(baseline),
        "official_units": len(before_flags),
        "stage2_correct": stage2_correct,
        "stage3_correct": stage3_correct,
        "aggregate_delta": stage3_correct - stage2_correct,
        "accuracy_delta": (stage3_correct - stage2_correct) / len(before_flags),
        "corrections": sum(
            not old and new for old, new in zip(before_flags, after_flags)
        ),
        "corruptions": sum(
            old and not new for old, new in zip(before_flags, after_flags)
        ),
        "oracle_fixes": oracle_fixes,
        "split_selections": selections,
        "selection_reasons": dict(sorted(reasons.items())),
        "rows": details,
    }


def select_policy(
    reports: Mapping[tuple[float, float], Mapping[str, Any]],
) -> tuple[float, float]:
    """Choose one global development policy, prioritizing cross-cell no-harm."""
    if not reports:
        raise ValueError("policy reports must be nonempty")

    def key(item: tuple[tuple[float, float], Mapping[str, Any]]) -> tuple[Any, ...]:
        policy, report = item
        deltas = report.get("cell_deltas")
        if not isinstance(deltas, Mapping) or not deltas:
            raise ValueError("policy report must expose cell deltas")
        aggregate = report.get("aggregate_delta")
        corruptions = report.get("corruptions")
        selections = report.get("split_selections")
        if not all(type(value) is int for value in (aggregate, corruptions, selections)):
            raise ValueError("policy report counts must be exact integers")
        values = tuple(deltas.values())
        if not all(type(value) is int for value in values):
            raise ValueError("cell deltas must be exact integers")
        return (
            all(value >= 0 for value in values),
            aggregate > 0,
            sum(value > 0 for value in values),
            aggregate,
            -corruptions,
            -selections,
            policy[0],
            policy[1],
        )

    return max(reports.items(), key=key)[0]


__all__ = [
    "candidate_outputs", "load_selected_calibrations", "official_correctness",
    "score_cell", "select_policy",
]
