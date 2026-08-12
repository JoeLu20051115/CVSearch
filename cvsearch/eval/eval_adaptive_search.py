"""Strict topic-level metrics for calibrated adaptive visual search."""

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


_FIELDS = frozenset({
    "row_id", "baseline_correct", "selected_correct", "support_label",
    "raw_support", "calibrated_support", "stopped", "action",
    "support_before", "support_after", "backtracked",
    "recovered_after_backtrack", "trajectory_length", "added_calls",
    "added_pixels", "added_latency_seconds", "baseline_rank_sha256",
    "selected_rank_sha256",
})
_ACTIONS = frozenset({"P0", "ZOOM", "EXPAND", "SPLIT", "BACKTRACK", "STOP"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _nonnegative_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _nonnegative_int(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    minimum = 1 if positive else 0
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _ece(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(10)]
    for probability, label in zip(probabilities, labels):
        bins[min(int(probability * 10), 9)].append((probability, label))
    total = len(probabilities)
    return sum(
        len(bucket) / total * abs(
            sum(item[0] for item in bucket) / len(bucket)
            - sum(item[1] for item in bucket) / len(bucket)
        )
        for bucket in bins if bucket
    )


def evaluate_adaptive_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate frozen decisions; this is the only boundary that sees outcomes."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("rows must be a nonempty sequence")
    snapshots = copy.deepcopy(list(rows))
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(snapshots):
        if type(row) is not dict or set(row) != _FIELDS:
            raise ValueError("adaptive evaluation row has an invalid exact schema")
        row_id = row["row_id"]
        if type(row_id) is not str or not row_id or row_id in seen:
            raise ValueError("row_id must be nonempty and unique")
        seen.add(row_id)
        for name in (
            "baseline_correct", "selected_correct", "stopped", "backtracked",
            "recovered_after_backtrack",
        ):
            if type(row[name]) is not bool:
                raise TypeError(f"{name} must be an exact boolean")
        if type(row["support_label"]) is not int or row["support_label"] not in {0, 1}:
            raise ValueError("support_label must be an exact binary integer")
        for name in (
            "raw_support", "calibrated_support", "support_before", "support_after",
        ):
            row[name] = _unit(row[name], name)
        row["trajectory_length"] = _nonnegative_int(
            row["trajectory_length"], "trajectory_length", positive=True,
        )
        row["added_calls"] = _nonnegative_int(row["added_calls"], "added_calls")
        row["added_pixels"] = _nonnegative_int(row["added_pixels"], "added_pixels")
        row["added_latency_seconds"] = _nonnegative_real(
            row["added_latency_seconds"], "added_latency_seconds",
        )
        if type(row["action"]) is not str or row["action"] not in _ACTIONS:
            raise ValueError("action is not canonical")
        for name in ("baseline_rank_sha256", "selected_rank_sha256"):
            if type(row[name]) is not str or _SHA256.fullmatch(row[name]) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if row["recovered_after_backtrack"] and not row["backtracked"]:
            raise ValueError("recovery requires a backtrack")
        normalized.append(row)

    total = len(normalized)
    baseline_correct = sum(row["baseline_correct"] for row in normalized)
    selected_correct = sum(row["selected_correct"] for row in normalized)
    corrections = sum(
        not row["baseline_correct"] and row["selected_correct"] for row in normalized
    )
    corruptions = sum(
        row["baseline_correct"] and not row["selected_correct"] for row in normalized
    )
    labels = [row["support_label"] for row in normalized]
    raw = [row["raw_support"] for row in normalized]
    calibrated = [row["calibrated_support"] for row in normalized]
    stopped = [row for row in normalized if row["stopped"]]
    false_stops = sum(row["support_label"] == 0 for row in stopped)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[row["action"]].append(row)
    actions = {}
    for action in sorted(grouped):
        action_rows = grouped[action]
        actions[action] = {
            "count": len(action_rows),
            "mean_support_gain": sum(
                row["support_after"] - row["support_before"] for row in action_rows
            ) / len(action_rows),
            "corrections": sum(
                not row["baseline_correct"] and row["selected_correct"]
                for row in action_rows
            ),
            "corruptions": sum(
                row["baseline_correct"] and not row["selected_correct"]
                for row in action_rows
            ),
        }

    backtracks = sum(row["backtracked"] for row in normalized)
    recoveries = sum(row["recovered_after_backtrack"] for row in normalized)
    rank_drifts = sum(
        row["baseline_rank_sha256"] != row["selected_rank_sha256"]
        for row in normalized
    )
    return {
        "schema_version": 1,
        "rows": total,
        "paired": {
            "baseline_correct": baseline_correct,
            "selected_correct": selected_correct,
            "baseline_accuracy": baseline_correct / total,
            "selected_accuracy": selected_correct / total,
            "accuracy_delta": (selected_correct - baseline_correct) / total,
            "corrections": corrections,
            "corruptions": corruptions,
        },
        "safety": {
            "stops": len(stopped),
            "false_stops": false_stops,
            "false_stop_rate": false_stops / len(stopped) if stopped else 0.0,
        },
        "calibration": {
            "raw_brier": sum((p - y) ** 2 for p, y in zip(raw, labels)) / total,
            "calibrated_brier": sum(
                (p - y) ** 2 for p, y in zip(calibrated, labels)
            ) / total,
            "raw_ece_10": _ece(raw, labels),
            "calibrated_ece_10": _ece(calibrated, labels),
        },
        "actions": actions,
        "trajectory": {
            "backtracks": backtracks,
            "recoveries": recoveries,
            "backtrack_recovery_rate": recoveries / backtracks if backtracks else 0.0,
        },
        "cost": {
            "total_added_calls": sum(row["added_calls"] for row in normalized),
            "mean_added_calls": sum(row["added_calls"] for row in normalized) / total,
            "total_added_pixels": sum(row["added_pixels"] for row in normalized),
            "mean_added_pixels": sum(row["added_pixels"] for row in normalized) / total,
            "total_added_latency_seconds": sum(
                row["added_latency_seconds"] for row in normalized
            ),
            "mean_added_latency_seconds": sum(
                row["added_latency_seconds"] for row in normalized
            ) / total,
            "mean_trajectory_length": sum(
                row["trajectory_length"] for row in normalized
            ) / total,
        },
        "phase1": {
            "rank_trace_drifts": rank_drifts,
            "all_rank_traces_preserved": rank_drifts == 0,
        },
    }


__all__ = ["evaluate_adaptive_rows"]
