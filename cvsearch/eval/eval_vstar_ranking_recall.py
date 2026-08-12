"""Evaluator-only V* ranking recall diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FAILURE_LABELS = (
    "quick_or_unranked",
    "planner_mismatch",
    "candidate_miss",
    "rank_miss",
    "answer_miss",
    "success",
)


def _xywh(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be an xywh sequence")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(number)
    if result[2] <= 0 or result[3] <= 0:
        raise ValueError(f"{name} width and height must be positive")
    return tuple(result)


def center_hit(target_xywh: Any, candidate_xywh: Any) -> bool:
    """Return whether the target center lies inside a candidate rectangle."""
    tx, ty, tw, th = _xywh(target_xywh, "target bbox")
    cx, cy, cw, ch = _xywh(candidate_xywh, "candidate bbox")
    target_x = tx + tw / 2.0
    target_y = ty + th / 2.0
    return cx <= target_x <= cx + cw and cy <= target_y <= cy + ch


def _event_key(detail: Mapping[str, Any]) -> tuple[Any, ...]:
    origin = detail.get("crop_origin")
    if not isinstance(origin, list) or len(origin) != 2:
        raise ValueError("candidate rank crop_origin must contain two values")
    return (
        detail.get("target"),
        detail.get("stage"),
        detail.get("tree_scope"),
        tuple(origin),
    )


def group_rank_events(details: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Split the append-only trace into consecutive ranker calls."""
    if isinstance(details, (str, bytes)) or not isinstance(details, Sequence):
        raise TypeError("candidate ranks must be a sequence")
    events: list[list[Mapping[str, Any]]] = []
    previous = object()
    for detail in details:
        if not isinstance(detail, Mapping):
            raise TypeError("candidate rank details must be mappings")
        key = _event_key(detail)
        if not events or key != previous:
            events.append([])
        events[-1].append(detail)
        previous = key
    return events


def _head_noun(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    tokens = re.findall(r"[A-Za-z]+", re.sub(r"['’]s\b", "", value.casefold()))
    return tokens[-1] if tokens else None


def _normalized_phrase(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    tokens = re.findall(r"[A-Za-z]+", re.sub(r"['’]s\b", "", value.casefold()))
    return " ".join(tokens) if tokens else None


def _targets(row: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    boxes = row.get("bbox")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("V* row bbox must be a nonempty list")
    return [_xywh(box, "target bbox") for box in boxes]


def _candidate_box(detail: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return _xywh(detail.get("bbox_original"), "candidate bbox")


def _target_names(row: Mapping[str, Any], count: int) -> list[str]:
    names = row.get("target_object")
    if (
        not isinstance(names, list)
        or len(names) != count
        or any(not isinstance(name, str) or not name.strip() for name in names)
    ):
        raise ValueError("target_object must align with evaluator target boxes")
    return names


def _matched_target_indices(runtime_target: Any, target_names: Sequence[str]) -> list[int]:
    runtime_phrase = _normalized_phrase(runtime_target)
    phrases = [_normalized_phrase(name) for name in target_names]
    exact = [index for index, phrase in enumerate(phrases) if phrase == runtime_phrase]
    if exact:
        return exact
    runtime_head = _head_noun(runtime_target)
    return [
        index for index, name in enumerate(target_names)
        if runtime_head is not None and _head_noun(name) == runtime_head
    ]


def _event_hit(
    event: Sequence[Mapping[str, Any]],
    targets: Sequence[tuple[float, float, float, float]],
    limit: int,
) -> bool:
    return any(
        center_hit(target, _candidate_box(detail))
        for detail in event[:limit]
        for target in targets
    )


def _concentration(
    events: Sequence[Sequence[Mapping[str, Any]]],
    targets: Sequence[tuple[float, float, float, float]],
    limit: int,
) -> float | None:
    values = []
    for event in events:
        for detail in event[:limit]:
            candidate = _candidate_box(detail)
            candidate_area = candidate[2] * candidate[3]
            for target in targets:
                if center_hit(target, candidate):
                    values.append((target[2] * target[3]) / candidate_area)
    return max(values) if values else None


def _summarize(rows: Sequence[Mapping[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    topic_hits = {k: [] for k in ks}
    all_target_topic_hits = {k: [] for k in ks}
    event_hits = {k: [] for k in ks}
    conditioned_event_hits = {k: [] for k in ks}
    pool_hits = []
    all_target_pool_hits = []
    concentrations = []
    ranked_topics = 0
    event_count = 0
    failures = Counter({label: 0 for label in FAILURE_LABELS})

    for row in rows:
        trace = row.get("method_trace")
        if not isinstance(trace, Mapping):
            raise ValueError("row method_trace must be a mapping")
        events = group_rank_events(trace.get("candidate_ranks", []))
        if not events:
            failures["quick_or_unranked"] += 1
            continue

        ranked_topics += 1
        event_count += len(events)
        targets = _targets(row)
        target_names = _target_names(row, len(targets))
        pool = any(_event_hit(event, targets, len(event)) for event in events)
        pool_hits.append(pool)
        all_target_pool_hits.append(all(
            any(
                center_hit(target, _candidate_box(detail))
                for event in events for detail in event
            )
            for target in targets
        ))
        for k in ks:
            hits = [_event_hit(event, targets, k) for event in events]
            event_hits[k].extend(hits)
            topic_hits[k].append(any(hits))
            all_target_topic_hits[k].append(all(
                any(
                    center_hit(target, _candidate_box(detail))
                    for event in events for detail in event[:k]
                )
                for target in targets
            ))
            for event in events:
                matched = _matched_target_indices(event[0].get("target"), target_names)
                if matched:
                    conditioned_event_hits[k].append(any(
                        center_hit(targets[index], _candidate_box(detail))
                        for detail in event[:k] for index in matched
                    ))
        concentration = _concentration(events, targets, 3)
        if concentration is not None:
            concentrations.append(concentration)

        evaluator_heads = {
            head for head in map(_head_noun, row.get("target_object", [])) if head is not None
        }
        runtime_heads = {
            head for head in (_head_noun(event[0].get("target")) for event in events)
            if head is not None
        }
        if evaluator_heads and evaluator_heads.isdisjoint(runtime_heads):
            failures["planner_mismatch"] += 1
        elif not pool:
            failures["candidate_miss"] += 1
        elif not topic_hits[3][-1]:
            failures["rank_miss"] += 1
        elif row.get("output") != 0:
            failures["answer_miss"] += 1
        else:
            failures["success"] += 1

    def ratio(values: Sequence[bool]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "ranked_topic_count": ranked_topics,
        "rank_event_count": event_count,
        "topic": {
            "recall_at": {str(k): ratio(topic_hits[k]) for k in ks},
            "all_required_targets_recall_at": {
                str(k): ratio(all_target_topic_hits[k]) for k in ks
            },
            "pool_upper_bound": ratio(pool_hits),
            "all_required_targets_pool_upper_bound": ratio(all_target_pool_hits),
            "median_top3_concentration": (
                statistics.median(concentrations) if concentrations else None
            ),
        },
        "event": {
            "recall_at": {str(k): ratio(event_hits[k]) for k in ks},
            "target_conditioned_recall_at": {
                str(k): ratio(conditioned_event_hits[k]) for k in ks
            },
            "target_conditioned_event_count": len(conditioned_event_hits[ks[0]]),
        },
        "failure_counts": {label: failures[label] for label in FAILURE_LABELS},
    }


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Evaluate trace order without exposing evaluator fields to inference."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("rows must be a nonempty sequence")
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError("rows must contain mappings")
    normalized_ks = tuple(ks)
    if (
        not normalized_ks
        or len(normalized_ks) != len(set(normalized_ks))
        or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in normalized_ks)
        or 3 not in normalized_ks
    ):
        raise ValueError("ks must contain unique positive integers including 3")

    summary = _summarize(rows, normalized_ks)
    test_types = sorted({row.get("test_type") for row in rows})
    if any(not isinstance(value, str) or not value for value in test_types):
        raise ValueError("test_type must be a nonempty string")
    summary.update({
        "schema_version": 1,
        "row_count": len(rows),
        "answer_accuracy": sum(row.get("output") == 0 for row in rows) / len(rows),
        "by_test_type": {
            test_type: _summarize(
                [row for row in rows if row.get("test_type") == test_type], normalized_ks
            )
            for test_type in test_types
        },
    })
    json.dumps(summary, allow_nan=False)
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError("JSONL input must not be empty")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.answers_file.read_bytes()
    rows = _load_jsonl(args.answers_file)
    report = evaluate_rows(rows)
    report["input"] = {
        "path": str(args.answers_file.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
