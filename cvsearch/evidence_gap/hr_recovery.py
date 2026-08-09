"""Fail-closed aggregate scoring for the locked HR-Bench recovery splits."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.baselines import reconstruct_quick_gate


_LOCKED_SPLITS = {
    "recovery-a": (
        2, 4, 5, 7, 8, 10, 13, 14, 19, 29, 32, 34, 35, 37, 40, 43, 46,
        62, 65, 67, 69, 81, 82, 85, 87, 90, 91, 97, 100, 102, 104, 108,
        111, 114, 115, 116, 118, 120, 122, 123, 124, 126, 128, 129, 131,
        132, 137, 138, 141, 143, 144, 146, 148, 150, 151, 154, 155, 157,
        164, 169, 174, 175, 183, 185, 187, 190, 191, 192, 196, 198, 199,
    ),
    "vault-b": (
        0, 3, 6, 11, 17, 18, 20, 26, 27, 31, 33, 36, 45, 47, 49, 50, 52,
        53, 55, 56, 57, 60, 61, 63, 64, 66, 72, 73, 74, 75, 77, 78, 79,
        88, 89, 92, 93, 94, 95, 96, 98, 99, 103, 107, 112, 113, 117, 119,
        121, 133, 135, 136, 139, 142, 145, 149, 156, 161, 162, 163, 165,
        170, 171, 172, 178, 180, 184, 189, 193, 195, 197,
    ),
}
_LOCKED_SPLIT_SHA256 = {
    "recovery-a": "d731fe78614b9165cb22deaf1956f2924d41e8c2be38edd5ed1fa477e9e9ae13",
    "vault-b": "ae6305b064ac93b861109297f906914870e20fe2a85289be6069855f32069c75",
}
_BOOTSTRAP_SAMPLES = 10_000
_BOOTSTRAP_SEED = 260_809
_WITHIN_RESOLUTION_FIELDS = ("question", "index", "category", "options", "answer")
_CROSS_RESOLUTION_FIELDS = ("question", "index", "category", "answer")
_BUDGET_FIELDS = (
    "mllm_calls", "max_mllm_calls", "processed_pixels", "max_processed_pixels"
)


def validate_locked_splits() -> None:
    if set(_LOCKED_SPLITS) != set(_LOCKED_SPLIT_SHA256):
        raise RuntimeError("locked recovery split registry is inconsistent")
    for name, ordinals in _LOCKED_SPLITS.items():
        encoded = ",".join(map(str, ordinals)).encode("ascii")
        if len(ordinals) != 71 or len(set(ordinals)) != 71:
            raise RuntimeError(f"locked recovery split {name} is not 71 unique topics")
        if hashlib.sha256(encoded).hexdigest() != _LOCKED_SPLIT_SHA256[name]:
            raise RuntimeError(f"locked recovery split {name} failed SHA-256 validation")


def locked_ordinals(name: str) -> tuple[int, ...]:
    try:
        return _LOCKED_SPLITS[name]
    except KeyError as error:
        raise ValueError(f"unknown recovery split: {name!r}") from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: Any, context: str) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be strict JSON") from error


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}")
            try:
                row = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_json_object,
                )
                json.dumps(row, allow_nan=False)
            except (json.JSONDecodeError, UnicodeError) as error:
                raise ValueError(f"invalid JSONL row at line {line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("JSONL input must not be empty")
    return rows


def _validate_hr_output(row: Mapping[str, Any], context: str) -> None:
    answers = row.get("answer")
    outputs = row.get("output")
    options = row.get("options")
    if (
        not isinstance(options, list)
        or len(options) != 4
        or any(not isinstance(option, str) for option in options)
    ):
        raise ValueError(f"{context} must contain four option strings")
    if (
        not isinstance(answers, list)
        or len(answers) != 4
        or any(answer not in ("A", "B", "C", "D") for answer in answers)
    ):
        raise ValueError(f"{context} must contain four A-D answers")
    if (
        not isinstance(outputs, list)
        or len(outputs) != 4
        or any(not isinstance(output, str) for output in outputs)
    ):
        raise ValueError(f"{context} must contain four string outputs")


def _topic_identity(
    row: Mapping[str, Any], context: str, fields: Sequence[str]
) -> tuple[Any, ...]:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{context} is missing topic fields: {missing}")
    return tuple(row[field] for field in fields)


def _validate_baseline_rows(
    direct_rows: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]],
    context: str,
) -> None:
    if len(direct_rows) != 200 or len(search_rows) != 200:
        raise ValueError(f"{context} retained baselines must each contain 200 rows")
    for ordinal, (direct, search) in enumerate(zip(direct_rows, search_rows)):
        if not isinstance(direct, Mapping) or not isinstance(search, Mapping):
            raise TypeError(f"{context} baseline rows must be JSON objects")
        _validate_hr_output(direct, f"{context} direct row {ordinal}")
        _validate_hr_output(search, f"{context} CVSearch row {ordinal}")
        if _topic_identity(direct, context, _WITHIN_RESOLUTION_FIELDS) != _topic_identity(
            search, context, _WITHIN_RESOLUTION_FIELDS
        ):
            raise ValueError(f"{context} baseline topic mismatch at ordinal {ordinal}")
        confidence = search.get("root_ans_conf")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not math.isfinite(float(confidence))
        ):
            raise ValueError(f"{context} CVSearch root_ans_conf must be finite")


def _validate_budget(
    trace: Mapping[str, Any], effective_config: Mapping[str, Any], context: str
) -> None:
    budget = trace.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError(f"{context} budget must be a mapping")
    missing = [field for field in _BUDGET_FIELDS if field not in budget]
    if missing:
        raise ValueError(f"{context} budget is missing fields: {missing}")
    for field in _BUDGET_FIELDS:
        value = budget[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{context} budget {field} must be a non-negative integer")
    if budget["mllm_calls"] > budget["max_mllm_calls"]:
        raise ValueError(f"{context} budget mllm_calls exceeds its maximum")
    if budget["processed_pixels"] > budget["max_processed_pixels"]:
        raise ValueError(f"{context} budget processed_pixels exceeds its maximum")
    for field in ("max_mllm_calls", "max_processed_pixels"):
        if effective_config.get(field) != budget[field]:
            raise ValueError(f"{context} budget {field} disagrees with effective config")


def _validate_method_rows(
    rows: Sequence[Mapping[str, Any]], expected: tuple[int, ...], context: str
) -> tuple[str, str, str, str]:
    if len(rows) != len(expected):
        raise ValueError(f"{context} method rows do not exactly cover the locked split")
    revisions: set[str] = set()
    config_ids: set[str] = set()
    fingerprints: set[str] = set()
    effective_configs: set[str] = set()
    for position, (row, expected_ordinal) in enumerate(zip(rows, expected)):
        if not isinstance(row, Mapping):
            raise TypeError(f"{context} method rows must be JSON objects")
        ordinal = row.get("_eg_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal != expected_ordinal:
            raise ValueError(f"{context} method ordinals do not exactly match the locked split")
        revision = row.get("_eg_code_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"{context} requires a nonempty code revision")
        revisions.add(revision)
        fingerprint = row.get("_eg_run_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError(f"{context} requires a nonempty run fingerprint")
        fingerprints.add(fingerprint)
        trace = row.get("method_trace")
        if not isinstance(trace, Mapping):
            raise ValueError(f"{context} method row {position} has no method trace")
        config_id = trace.get("config_id")
        if not isinstance(config_id, str) or not config_id.strip():
            raise ValueError(f"{context} requires a nonempty config_id")
        config_ids.add(config_id)
        effective_config = trace.get("effective_config")
        if not isinstance(effective_config, Mapping):
            raise ValueError(f"{context} effective config must be a mapping")
        effective_config = dict(effective_config)
        if effective_config.get("config_id") != config_id:
            raise ValueError(f"{context} effective config_id must equal trace config_id")
        effective_configs.add(_canonical_json(effective_config, f"{context} effective config"))
        _validate_budget(trace, effective_config, context)
        if trace.get("budget_interrupted") is not False:
            raise ValueError(f"{context} contains a budget interruption")
        _validate_hr_output(row, f"{context} method row {position}")
        _topic_identity(row, context, _WITHIN_RESOLUTION_FIELDS)
    if len(revisions) != 1:
        raise ValueError(f"{context} must use exactly one code revision")
    if len(config_ids) != 1:
        raise ValueError(f"{context} must use exactly one config_id")
    if len(fingerprints) != 1:
        raise ValueError(f"{context} must use exactly one run fingerprint")
    if len(effective_configs) != 1:
        raise ValueError(f"{context} must use exactly one effective config")
    return (
        next(iter(revisions)), next(iter(config_ids)), next(iter(fingerprints)),
        next(iter(effective_configs)),
    )


def _predicted_letter(output: str) -> str:
    if len(output) == 1:
        return output
    return next((letter for letter in output if letter in ("A", "B", "C", "D")), "")


def _topic_accuracy(row: Mapping[str, Any]) -> float:
    return sum(
        answer == _predicted_letter(output)
        for answer, output in zip(row["answer"], row["output"])
    ) / 4


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _bootstrap_intervals(
    deltas_4k: Sequence[float], deltas_8k: Sequence[float]
) -> tuple[float, float, float, float]:
    if len(deltas_4k) != len(deltas_8k) or not deltas_4k:
        raise ValueError("paired topic deltas must have equal nonzero length")
    randomizer = random.Random(_BOOTSTRAP_SEED)
    n_topics = len(deltas_4k)
    samples_4k = []
    samples_8k = []
    interactions = []
    for _ in range(_BOOTSTRAP_SAMPLES):
        indices = [randomizer.randrange(n_topics) for _ in range(n_topics)]
        delta_4k = sum(deltas_4k[index] for index in indices) / n_topics
        delta_8k = sum(deltas_8k[index] for index in indices) / n_topics
        samples_4k.append(delta_4k)
        samples_8k.append(delta_8k)
        interactions.append(delta_8k - delta_4k)
    samples_4k.sort()
    samples_8k.sort()
    interactions.sort()
    return (
        _percentile(samples_4k, 0.05),
        _percentile(samples_8k, 0.05),
        _percentile(interactions, 0.025),
        _percentile(interactions, 0.975),
    )


def score_recovery(
    split_name: str,
    *,
    method_4k: Sequence[Mapping[str, Any]],
    method_8k: Sequence[Mapping[str, Any]],
    direct_4k: Sequence[Mapping[str, Any]],
    cvsearch_4k: Sequence[Mapping[str, Any]],
    direct_8k: Sequence[Mapping[str, Any]],
    cvsearch_8k: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and score one explicitly selected locked recovery split."""
    expected = locked_ordinals(split_name)
    revision_4k, config_4k, _, effective_config_4k = _validate_method_rows(
        method_4k, expected, "4K"
    )
    revision_8k, config_8k, _, effective_config_8k = _validate_method_rows(
        method_8k, expected, "8K"
    )
    if revision_4k != revision_8k:
        raise ValueError("4K and 8K must use the same code revision")
    if config_4k != config_8k:
        raise ValueError("4K and 8K must use the same config_id")
    if effective_config_4k != effective_config_8k:
        raise ValueError("4K and 8K must use the same effective config")
    _validate_baseline_rows(direct_4k, cvsearch_4k, "4K")
    _validate_baseline_rows(direct_8k, cvsearch_8k, "8K")

    baseline_4k = reconstruct_quick_gate(direct_4k, cvsearch_4k, 0.6)
    baseline_8k = reconstruct_quick_gate(direct_8k, cvsearch_8k, 0.6)
    method_scores = {"4K": [], "8K": []}
    baseline_scores = {"4K": [], "8K": []}
    for index, ordinal in enumerate(expected):
        method_rows = (method_4k[index], method_8k[index])
        baseline_rows = (baseline_4k[ordinal], baseline_8k[ordinal])
        for resolution, method_row, baseline_row in zip(
            ("4K", "8K"), method_rows, baseline_rows
        ):
            if _topic_identity(
                method_row, resolution, _WITHIN_RESOLUTION_FIELDS
            ) != _topic_identity(baseline_row, resolution, _WITHIN_RESOLUTION_FIELDS):
                raise ValueError(f"{resolution} topic identity mismatch at ordinal {ordinal}")
        if _topic_identity(
            method_rows[0], "4K", _CROSS_RESOLUTION_FIELDS
        ) != _topic_identity(method_rows[1], "8K", _CROSS_RESOLUTION_FIELDS):
            raise ValueError(f"paired topic identity mismatch at ordinal {ordinal}")
        method_scores["4K"].append(_topic_accuracy(method_4k[index]))
        method_scores["8K"].append(_topic_accuracy(method_8k[index]))
        baseline_scores["4K"].append(_topic_accuracy(baseline_4k[ordinal]))
        baseline_scores["8K"].append(_topic_accuracy(baseline_8k[ordinal]))

    deltas = {
        resolution: [method - baseline for method, baseline in zip(
            method_scores[resolution], baseline_scores[resolution]
        )]
        for resolution in ("4K", "8K")
    }
    lower_4k, lower_8k, interaction_low, interaction_high = _bootstrap_intervals(
        deltas["4K"], deltas["8K"]
    )
    delta_4k = _mean(deltas["4K"])
    delta_8k = _mean(deltas["8K"])
    return {
        "n_topics": len(expected),
        "hr-bench_4k": {
            "method_accuracy": _mean(method_scores["4K"]),
            "baseline_accuracy": _mean(baseline_scores["4K"]),
            "delta": delta_4k,
            "one_sided_95_lower": lower_4k,
        },
        "hr-bench_8k": {
            "method_accuracy": _mean(method_scores["8K"]),
            "baseline_accuracy": _mean(baseline_scores["8K"]),
            "delta": delta_8k,
            "one_sided_95_lower": lower_8k,
        },
        "interaction_8k_minus_4k": {
            "delta": delta_8k - delta_4k,
            "two_sided_95_ci": [interaction_low, interaction_high],
        },
        "point_gate_pass": delta_4k > 0 and delta_8k > 0,
    }


validate_locked_splits()
