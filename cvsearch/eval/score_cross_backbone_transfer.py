"""Compact scoring helpers for paired Phase-1/Phase-2 transfer runs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from cvsearch.evidence_gap.answers import official_letter
from cvsearch.evidence_gap.types import EXPAND, ZOOM

from .replay_adaptive_search import FrozenSelectedCalibration
from .treebench_support_proxy import treebench_geometry_support_label
from .vstar_support_proxy import (
    visible_crops_from_audit,
    vstar_geometry_support_label,
)


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1]")
    return result


def _ece_10(predictions: Sequence[float], labels: Sequence[int]) -> float:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for prediction, label in zip(predictions, labels):
        buckets[min(int(prediction * 10), 9)].append((prediction, label))
    total = len(predictions)
    return sum(
        len(bucket) / total * abs(
            sum(prediction for prediction, _ in bucket) / len(bucket)
            - sum(label for _, label in bucket) / len(bucket)
        )
        for bucket in buckets if bucket
    )


def _auroc(predictions: Sequence[float], labels: Sequence[int]) -> float | None:
    positive = [p for p, label in zip(predictions, labels) if label == 1]
    negative = [p for p, label in zip(predictions, labels) if label == 0]
    if not positive or not negative:
        return None
    favorable = 0.0
    for pos in positive:
        for neg in negative:
            favorable += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return favorable / (len(positive) * len(negative))


def calibration_metrics(
    rows: Sequence[Mapping[str, Any]],
    calibration: FrozenSelectedCalibration,
) -> dict[str, Any]:
    """Score frozen support calibration on rows excluded from fitting."""
    if not isinstance(calibration, FrozenSelectedCalibration):
        raise TypeError("calibration must be a frozen selected calibration")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("calibration evaluation rows must be nonempty")
    raw: list[float] = []
    labels: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("calibration evaluation rows must be mappings")
        raw.append(_unit(row.get("raw_support"), "raw_support"))
        label = row.get("support_sufficient")
        if type(label) is not int or label not in {0, 1}:
            raise ValueError("support_sufficient must be an exact binary integer")
        labels.append(label)
    calibrated = [calibration.predict(value) for value in raw]
    return {
        "samples": len(rows),
        "positive": sum(labels),
        "raw_brier": sum((p - y) ** 2 for p, y in zip(raw, labels)) / len(rows),
        "calibrated_brier": sum(
            (p - y) ** 2 for p, y in zip(calibrated, labels)
        ) / len(rows),
        "raw_ece_10": _ece_10(raw, labels),
        "calibrated_ece_10": _ece_10(calibrated, labels),
        "raw_auroc": _auroc(raw, labels),
        "calibrated_auroc": _auroc(calibrated, labels),
    }


def extract_support_rows(
    benchmark: str, rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract answer-free support/geometry pairs from successful scale probes."""
    if benchmark not in {"vstar", "treebench"}:
        raise ValueError("support geometry is available only for V* and TreeBench")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("observation rows must be a sequence")
    extracted: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("observation rows must contain mappings")
        ordinal = row.get("_eg_ordinal")
        source_group = row.get("input_image")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("observation ordinal must be a nonnegative integer")
        if type(source_group) is not str or not source_group:
            raise ValueError("input_image must be a nonempty source group")
        trace = row.get("method_trace")
        steps = trace.get("steps") if isinstance(trace, Mapping) else None
        if not isinstance(steps, list):
            continue
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping) or step.get("action") not in {ZOOM, EXPAND}:
                continue
            action = step["action"]
            audit = step.get("zoom_audit" if action == ZOOM else "expand_audit")
            if not isinstance(audit, Mapping):
                continue
            batch = audit.get("batch_result")
            if not isinstance(batch, Mapping) or batch.get("status") != "success":
                continue
            current = audit.get("current_gap_support")
            candidate = audit.get("candidate_gap_support")
            if not isinstance(current, Mapping) or not isinstance(candidate, Mapping):
                continue
            current_crops, candidate_crops = visible_crops_from_audit(action, audit)
            labeler = (
                vstar_geometry_support_label
                if benchmark == "vstar" else treebench_geometry_support_label
            )
            for view, support, crops in (
                ("current", current, current_crops),
                ("candidate", candidate, candidate_crops),
            ):
                extracted.append({
                    "row_id": f"{ordinal}:{step_index}:{action}:{view}",
                    "source_group": source_group,
                    "raw_support": _unit(support.get("p_yes"), "support p_yes"),
                    "support_sufficient": labeler(row, crops),
                })
    return extracted


def _paired_by_ordinal(
    baseline_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    def index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[int, Mapping[str, Any]]:
        result: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError(f"{name} rows must be mappings")
            ordinal = row.get("_eg_ordinal")
            if type(ordinal) is not int or ordinal < 0 or ordinal in result:
                raise ValueError(f"{name} ordinals must be unique nonnegative integers")
            result[ordinal] = row
        return result

    baseline = index(baseline_rows, "baseline")
    selected = index(selected_rows, "selected")
    if not baseline or baseline.keys() != selected.keys():
        raise ValueError("paired datasets must contain identical nonempty ordinals")
    return [(baseline[key], selected[key]) for key in sorted(baseline)]


def _official_correctness(
    benchmark: str, row: Mapping[str, Any], output: Any,
) -> tuple[list[bool], int]:
    if benchmark == "vstar":
        return [output == 0], 1
    if benchmark in {"hr-bench_4k", "hr-bench_8k"}:
        answers = row.get("answer")
        if type(answers) is not list or type(output) is not list:
            raise ValueError("HR rows require answer and output lists")
        if len(answers) != len(output) or not answers:
            raise ValueError("HR answer and output lists must align")
        parsed = [official_letter(value) for value in output]
        return [prediction == answer for prediction, answer in zip(parsed, answers)], sum(
            prediction is not None for prediction in parsed
        )
    if benchmark == "treebench":
        if not isinstance(output, str):
            return [False], 0
        prediction = output.strip().upper()
        answer = str(row.get("answer", "")).strip().upper()
        return [prediction == answer], int(official_letter(prediction) is not None)
    raise ValueError("unsupported transfer benchmark")


def score_paired_dataset(
    benchmark: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score frozen replay outputs in each benchmark's official unit."""
    pairs = _paired_by_ordinal(baseline_rows, selected_rows)
    baseline_flags: list[bool] = []
    selected_flags: list[bool] = []
    parseable = 0
    actions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for baseline, selected in pairs:
        source = selected.get("selected_source")
        reason = selected.get("reason")
        if type(source) is not str or not source:
            raise ValueError("selected_source must be a nonempty string")
        if type(reason) is not str or not reason:
            raise ValueError("selection reason must be a nonempty string")
        actions[source] += 1
        reasons[reason] += 1
        before, _ = _official_correctness(
            benchmark, baseline, baseline.get("output"),
        )
        after, parsed = _official_correctness(
            benchmark, baseline, selected.get("selected_output"),
        )
        if len(before) != len(after):
            raise AssertionError("paired official-unit counts drifted")
        baseline_flags.extend(before)
        selected_flags.extend(after)
        parseable += parsed
    total = len(baseline_flags)
    baseline_correct = sum(baseline_flags)
    selected_correct = sum(selected_flags)
    result = {
        "topics": len(pairs),
        "official_units": total,
        "baseline_correct": baseline_correct,
        "selected_correct": selected_correct,
        "baseline_accuracy": baseline_correct / total,
        "selected_accuracy": selected_correct / total,
        "accuracy_delta": (selected_correct - baseline_correct) / total,
        "corrections": sum(
            not before and after
            for before, after in zip(baseline_flags, selected_flags)
        ),
        "corruptions": sum(
            before and not after
            for before, after in zip(baseline_flags, selected_flags)
        ),
        "selected_actions": dict(sorted(actions.items())),
        "selection_reasons": dict(sorted(reasons.items())),
    }
    if benchmark == "treebench":
        result["parseable_outputs"] = parseable
    return result


__all__ = [
    "calibration_metrics", "extract_support_rows", "score_paired_dataset",
]
