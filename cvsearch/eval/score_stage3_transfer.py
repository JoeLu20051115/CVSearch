"""Evaluator-only paired metrics for Stage 3 split search transfer."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


_FIELDS = frozenset({
    "row_id", "backbone", "dataset", "stage2_correct", "stage3_correct",
    "candidate_correct", "selected_source", "raw_support",
    "calibrated_support", "support_sufficient", "trajectory_s", "calls",
    "pixels", "latency_seconds", "branch_count", "max_depth", "backtracks",
})
_DATASETS = frozenset({
    "vstar", "hr_bench_4k", "hr_bench_8k", "treebench",
})


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _count(value: Any, name: str, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its exact count range")
    return value


def _flags(value: Any, name: str) -> tuple[bool, ...]:
    if type(value) is not tuple or not value or not all(type(item) is bool for item in value):
        raise ValueError(f"{name} must be a nonempty exact boolean tuple")
    return value


def _normalize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("Stage-3 evaluation rows must be a nonempty sequence")
    seen = set()
    normalized = []
    for row in rows:
        if type(row) is not dict or set(row) != _FIELDS:
            raise ValueError("Stage-3 evaluation rows must use the exact evaluator schema")
        row_id = row["row_id"]
        backbone = row["backbone"]
        dataset = row["dataset"]
        if type(row_id) is not str or not row_id or row_id in seen:
            raise ValueError("Stage-3 row_id must be unique and nonempty")
        seen.add(row_id)
        if type(backbone) is not str or not backbone:
            raise ValueError("Stage-3 backbone must be nonempty")
        if dataset not in _DATASETS:
            raise ValueError("Stage-3 dataset is outside the declared transfer scope")
        before = _flags(row["stage2_correct"], "stage2_correct")
        after = _flags(row["stage3_correct"], "stage3_correct")
        if len(before) != len(after):
            raise ValueError("Stage-2 and Stage-3 official units must align")
        candidates = row["candidate_correct"]
        if type(candidates) is not tuple or not all(
            type(candidate) is tuple and len(candidate) == len(before)
            and all(type(item) is bool for item in candidate)
            for candidate in candidates
        ):
            raise ValueError("candidate correctness must align with official units")
        if row["selected_source"] not in {"P0", "SPLIT"}:
            raise ValueError("Stage-3 selected source must be P0 or SPLIT")
        trajectory = row["trajectory_s"]
        if trajectory is not None and (
            type(trajectory) is not int or not -3 <= trajectory <= 3
        ):
            raise ValueError("trajectory_s must be a bounded exact integer or null")
        branch_count = _count(row["branch_count"], "branch_count", 4)
        max_depth = _count(row["max_depth"], "max_depth", 2)
        backtracks = _count(row["backtracks"], "backtracks", 3)
        if row["selected_source"] == "SPLIT" and (
            branch_count == 0 or max_depth != 2 or trajectory is None
        ):
            raise ValueError("selected SPLIT lacks bounded search provenance")
        normalized.append({
            **row,
            "stage2_correct": before,
            "stage3_correct": after,
            "candidate_correct": candidates,
            "raw_support": _unit(row["raw_support"], "raw_support"),
            "calibrated_support": _unit(
                row["calibrated_support"], "calibrated_support",
            ),
            "support_sufficient": _count(
                row["support_sufficient"], "support_sufficient", 1,
            ),
            "calls": _count(row["calls"], "calls"),
            "pixels": _count(row["pixels"], "pixels"),
            "latency_seconds": _nonnegative(
                row["latency_seconds"], "latency_seconds",
            ),
            "branch_count": branch_count,
            "max_depth": max_depth,
            "backtracks": backtracks,
        })
    return normalized


def _paired_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    before = [flag for row in rows for flag in row["stage2_correct"]]
    after = [flag for row in rows for flag in row["stage3_correct"]]
    total = len(before)
    baseline = sum(before)
    selected = sum(after)
    return {
        "topics": len(rows),
        "official_units": total,
        "stage2_correct": baseline,
        "stage3_correct": selected,
        "stage2_accuracy": baseline / total,
        "stage3_accuracy": selected / total,
        "accuracy_delta": (selected - baseline) / total,
        "corrections": sum(not old and new for old, new in zip(before, after)),
        "corruptions": sum(old and not new for old, new in zip(before, after)),
        "selected_sources": dict(sorted(Counter(
            row["selected_source"] for row in rows
        ).items())),
    }


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], key: str,
) -> dict[str, dict[str, Any]]:
    groups = sorted({row[key] for row in rows})
    return {
        group: _paired_metrics([row for row in rows if row[key] == group])
        for group in groups
    }


def _ece_10(predictions: Sequence[float], labels: Sequence[int]) -> float:
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for prediction, label in zip(predictions, labels):
        buckets[min(int(prediction * 10), 9)].append((prediction, label))
    return sum(
        len(bucket) / len(predictions) * abs(
            sum(prediction for prediction, _ in bucket) / len(bucket)
            - sum(label for _, label in bucket) / len(bucket)
        )
        for bucket in buckets if bucket
    )


def _auroc(predictions: Sequence[float], labels: Sequence[int]) -> float | None:
    positive = [value for value, label in zip(predictions, labels) if label]
    negative = [value for value, label in zip(predictions, labels) if not label]
    if not positive or not negative:
        return None
    return sum(
        1.0 if high > low else 0.5 if high == low else 0.0
        for high in positive for low in negative
    ) / (len(positive) * len(negative))


def _calibration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw = [row["raw_support"] for row in rows]
    calibrated = [row["calibrated_support"] for row in rows]
    labels = [row["support_sufficient"] for row in rows]
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


def evaluate_stage3_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate frozen Stage-3 decisions after labels are opened."""
    normalized = _normalize(rows)
    cells: dict[str, dict[str, Any]] = {}
    for backbone in sorted({row["backbone"] for row in normalized}):
        cells[backbone] = {}
        for dataset in sorted({
            row["dataset"] for row in normalized if row["backbone"] == backbone
        }):
            cells[backbone][dataset] = _paired_metrics([
                row for row in normalized
                if row["backbone"] == backbone and row["dataset"] == dataset
            ])
    trajectory = Counter(
        "missing" if row["trajectory_s"] is None
        else "positive" if row["trajectory_s"] > 0
        else "negative" if row["trajectory_s"] < 0 else "plateau"
        for row in normalized
    )
    return {
        "cells": cells,
        "by_backbone": _group_metrics(normalized, "backbone"),
        "by_dataset": _group_metrics(normalized, "dataset"),
        "aggregate": _paired_metrics(normalized),
        "calibration": _calibration(normalized),
        "trajectory": {
            name: trajectory.get(name, 0)
            for name in ("positive", "plateau", "negative", "missing")
        },
        "cost": {
            "total_calls": sum(row["calls"] for row in normalized),
            "total_pixels": sum(row["pixels"] for row in normalized),
            "total_latency_seconds": sum(
                row["latency_seconds"] for row in normalized
            ),
            "mean_calls": sum(row["calls"] for row in normalized) / len(normalized),
            "mean_pixels": sum(row["pixels"] for row in normalized) / len(normalized),
            "mean_latency_seconds": sum(
                row["latency_seconds"] for row in normalized
            ) / len(normalized),
            "max_branch_count": max(row["branch_count"] for row in normalized),
            "max_depth": max(row["max_depth"] for row in normalized),
            "total_backtracks": sum(row["backtracks"] for row in normalized),
        },
    }


def _oracle_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = fixes = 0
    for row in rows:
        for index, baseline in enumerate(row["stage2_correct"]):
            if baseline:
                continue
            errors += 1
            fixes += any(candidate[index] for candidate in row["candidate_correct"])
    return {"official_units": sum(len(row["stage2_correct"]) for row in rows),
            "baseline_errors": errors, "oracle_fixes": fixes}


def split_candidate_oracle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure candidate-generation ceiling independently of the selector."""
    normalized = _normalize(rows)
    cells: dict[str, dict[str, Any]] = {}
    for backbone in sorted({row["backbone"] for row in normalized}):
        cells[backbone] = {}
        for dataset in sorted({
            row["dataset"] for row in normalized if row["backbone"] == backbone
        }):
            cells[backbone][dataset] = _oracle_metrics([
                row for row in normalized
                if row["backbone"] == backbone and row["dataset"] == dataset
            ])
    return {
        "cells": cells,
        "by_backbone": {
            value: _oracle_metrics([row for row in normalized if row["backbone"] == value])
            for value in sorted({row["backbone"] for row in normalized})
        },
        "by_dataset": {
            value: _oracle_metrics([row for row in normalized if row["dataset"] == value])
            for value in sorted({row["dataset"] for row in normalized})
        },
        "aggregate": _oracle_metrics(normalized),
    }


__all__ = ["evaluate_stage3_rows", "split_candidate_oracle"]
