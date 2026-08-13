#!/usr/bin/env python3
"""Freeze, generate, and score CPU-only unified-selector replays."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cvsearch.eval.analyze_split_search import (
    candidate_outputs,
    load_selected_calibrations,
    official_correctness,
)
from cvsearch.eval.replay_split_search import _split_audit

from .freeze_uncertainty_support import (
    BENCHMARKS,
    DevelopmentRecord,
    canonical_payload_hash,
    freeze_policy,
    source_group,
)
from .replay_uncertainty_support import (
    UnifiedPolicy,
    replay_uncertainty_support,
)


EXPECTED_BACKBONES = ("internvl", "qwen")
EXPECTED_BENCHMARKS = (
    "hr_bench_4k", "hr_bench_8k", "treebench", "vstar",
)
EXPECTED_CELLS = frozenset(
    f"{backbone}/{benchmark}"
    for backbone in EXPECTED_BACKBONES
    for benchmark in EXPECTED_BENCHMARKS
)
_INFERENCE_ROW_FIELDS = frozenset({
    "_eg_ordinal", "answer_type", "options", "output", "method_trace",
})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _artifact_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return _hash_value(unsigned)


def _sanitize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if type(row) is not dict:
        raise TypeError("selector input rows must be exact dictionaries")
    result = {
        key: copy.deepcopy(row[key])
        for key in _INFERENCE_ROW_FIELDS if key in row
    }
    if set(result) != _INFERENCE_ROW_FIELDS:
        raise ValueError("selector input row lacks a required inference field")
    return result


def _index_rows(
    rows: Sequence[Mapping[str, Any]], name: str,
) -> dict[int, Mapping[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError(f"{name} rows must be a sequence")
    result = {}
    for row in rows:
        if type(row) is not dict:
            raise TypeError(f"{name} rows must be exact dictionaries")
        ordinal = row.get("_eg_ordinal")
        if type(ordinal) is not int or ordinal < 0 or ordinal in result:
            raise ValueError(f"{name} ordinals must be unique nonnegative integers")
        result[ordinal] = row
    if not result:
        raise ValueError(f"{name} rows must be nonempty")
    return result


def _split_cell(cell: str) -> tuple[str, str]:
    if not isinstance(cell, str) or cell.count("/") != 1:
        raise ValueError("cell keys must be backbone/benchmark")
    backbone, benchmark = cell.split("/")
    if not backbone or benchmark not in BENCHMARKS:
        raise ValueError("cell key is unsupported")
    return backbone, benchmark


def _fixed_observation_contract(row: Mapping[str, Any]) -> bool:
    audit = _split_audit(row)
    if not isinstance(audit, Mapping) or not (
        audit.get("max_depth") == 2
        and audit.get("max_observed_branches") == 6
        and audit.get("max_screening_probes") == 16
        and audit.get("render_policy")
        == "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
    ):
        return False
    probes = audit.get("screening_probes")
    roots = audit.get("root_ranked_siblings")
    branches = audit.get("branches")
    if not all(isinstance(value, list) for value in (probes, roots, branches)):
        return False
    no_op_reason = audit.get("no_op_reason")
    if no_op_reason == "split_invalid_evidence_requirements":
        return not probes and not roots and not branches
    return (
        no_op_reason is None
        and len(probes) == 16
        and len(roots) == 4
        and len(branches) == 6
    )


def generate_decisions(
    stage2_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    split_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Any],
    policy: UnifiedPolicy,
) -> dict[str, Any]:
    """Generate a hash-bound decision artifact without evaluator labels."""
    if not isinstance(policy, UnifiedPolicy):
        raise TypeError("decision policy must be a frozen unified policy")
    if not isinstance(stage2_by_cell, Mapping) or not isinstance(split_by_cell, Mapping):
        raise TypeError("decision suites must be mappings")
    if set(stage2_by_cell) != set(split_by_cell) or not stage2_by_cell:
        raise ValueError("Stage-2 and SPLIT decision cells must align exactly")
    decisions = []
    inputs = {}
    support_hashes = {}
    for cell in sorted(stage2_by_cell):
        backbone, _ = _split_cell(cell)
        calibration = calibrations.get(backbone)
        if calibration is None:
            raise ValueError(f"missing frozen support calibration for {backbone}")
        support_hashes[backbone] = calibration.manifest_sha256
        stage2 = _index_rows(stage2_by_cell[cell], f"{cell} Stage-2")
        split = _index_rows(split_by_cell[cell], f"{cell} SPLIT")
        if stage2.keys() != split.keys():
            raise ValueError(f"{cell} Stage-2 and SPLIT ordinals differ")
        sanitized_stage2 = {
            ordinal: _sanitize_row(stage2[ordinal]) for ordinal in sorted(stage2)
        }
        sanitized_split = {
            ordinal: _sanitize_row(split[ordinal]) for ordinal in sorted(split)
        }
        inputs[cell] = {
            "rows": len(stage2),
            "stage2_observations_sha256": _hash_value(sanitized_stage2),
            "split_observations_sha256": _hash_value(sanitized_split),
            "fixed_observations": all(
                _fixed_observation_contract(row)
                for row in sanitized_split.values()
            ),
        }
        for ordinal in sorted(stage2):
            decision = replay_uncertainty_support(
                sanitized_stage2[ordinal], sanitized_split[ordinal],
                calibration, policy,
            )
            record = {
                "cell": cell,
                "ordinal": ordinal,
                "stage2_selected_output": copy.deepcopy(
                    decision["stage2_selected_output"],
                ),
                "selected_output": copy.deepcopy(decision["selected_output"]),
                "selected_source": decision["selected_source"],
                "reason": decision["reason"],
                "selected_branch": decision["selected_branch"],
                "used_backtrack": decision["used_backtrack"],
                "observations": decision["observations"],
                "transitions": copy.deepcopy(decision["transitions"]),
                "candidate_outputs": [
                    copy.deepcopy(output)
                    for output in candidate_outputs(sanitized_split[ordinal])
                ],
            }
            if "failure_detail" in decision:
                record["failure_detail"] = decision["failure_detail"]
            decisions.append(record)
    payload = {
        "schema_version": 1,
        "artifact_kind": "unified-uncertainty-support-decisions",
        "data_scope": "label_blind_locked_replay",
        "policy_sha256": policy.payload_sha256,
        "support_calibrations": dict(sorted(support_hashes.items())),
        "inputs": inputs,
        "decisions": decisions,
    }
    payload["decision_payload_sha256"] = _artifact_hash(
        payload, "decision_payload_sha256",
    )
    return payload


def _pool_cells(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    by: str,
) -> dict[str, dict[str, Any]]:
    pooled: dict[str, Counter[str]] = {}
    for cell, metrics in cells.items():
        backbone, benchmark = _split_cell(cell)
        key = backbone if by == "backbone" else benchmark
        bucket = pooled.setdefault(key, Counter())
        for name in (
            "topics", "official_units", "baseline_correct", "selected_correct",
            "corrections", "corruptions", "selections", "observations",
            "oracle_fixes",
        ):
            bucket[name] += metrics[name]
    result = {}
    for key, bucket in sorted(pooled.items()):
        values = dict(bucket)
        values["delta"] = values["selected_correct"] - values["baseline_correct"]
        values["baseline_accuracy"] = (
            values["baseline_correct"] / values["official_units"]
        )
        values["selected_accuracy"] = (
            values["selected_correct"] / values["official_units"]
        )
        values["accuracy_delta"] = values["delta"] / values["official_units"]
        values["mean_observations"] = values["observations"] / values["topics"]
        result[key] = values
    return result


def score_decisions(
    labeled_stage2_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    artifact: Mapping[str, Any],
    *,
    expected_topics: int = 112,
    expected_units: int = 256,
) -> dict[str, Any]:
    """Score immutable decisions in official units after replay is frozen."""
    if not isinstance(artifact, Mapping):
        raise TypeError("decision artifact must be a mapping")
    if artifact.get("decision_payload_sha256") != _artifact_hash(
        artifact, "decision_payload_sha256",
    ):
        raise ValueError("decision artifact hash mismatch")
    raw_decisions = artifact.get("decisions")
    inputs = artifact.get("inputs")
    if not isinstance(raw_decisions, list) or not isinstance(inputs, Mapping):
        raise ValueError("decision artifact is incomplete")
    indexed_decisions = {}
    for decision in raw_decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("decision record is invalid")
        key = (decision.get("cell"), decision.get("ordinal"))
        if key in indexed_decisions:
            raise ValueError("decision records must be unique")
        indexed_decisions[key] = decision

    cells = {}
    action_counts: Counter[str] = Counter()
    for cell in sorted(labeled_stage2_by_cell):
        _, benchmark = _split_cell(cell)
        rows = _index_rows(labeled_stage2_by_cell[cell], f"{cell} scoring")
        sanitized = {
            ordinal: _sanitize_row(rows[ordinal]) for ordinal in sorted(rows)
        }
        binding = inputs.get(cell)
        if (
            not isinstance(binding, Mapping)
            or binding.get("rows") != len(rows)
            or binding.get("stage2_observations_sha256") != _hash_value(sanitized)
        ):
            raise ValueError(f"{cell} Stage-2 scoring input hash mismatch")
        before_flags = []
        after_flags = []
        corrections = corruptions = selections = observations = oracle_fixes = 0
        for ordinal in sorted(rows):
            decision = indexed_decisions.pop((cell, ordinal), None)
            if not isinstance(decision, Mapping):
                raise ValueError(f"{cell}/{ordinal} decision is missing")
            before = official_correctness(
                benchmark, rows[ordinal], decision.get("stage2_selected_output"),
            )
            after = official_correctness(
                benchmark, rows[ordinal], decision.get("selected_output"),
            )
            candidates = tuple(
                official_correctness(benchmark, rows[ordinal], output)
                for output in decision.get("candidate_outputs", ())
            )
            if len(before) != len(after) or any(
                len(candidate) != len(before) for candidate in candidates
            ):
                raise ValueError("decision official-unit accounting drifted")
            before_flags.extend(before)
            after_flags.extend(after)
            corrections += sum(not old and new for old, new in zip(before, after))
            corruptions += sum(old and not new for old, new in zip(before, after))
            oracle_fixes += sum(
                not old and any(candidate[index] for candidate in candidates)
                for index, old in enumerate(before)
            )
            selections += decision.get("selected_source") == "SPLIT"
            count = decision.get("observations")
            if type(count) is not int or count < 0:
                raise ValueError("decision observation count is invalid")
            observations += count
            transitions = decision.get("transitions")
            if not isinstance(transitions, list):
                raise ValueError("decision transition trace is invalid")
            for transition in transitions:
                action = transition.get("action") if isinstance(transition, Mapping) else None
                if not isinstance(action, str):
                    raise ValueError("decision transition action is invalid")
                action_counts[action] += 1
        baseline_correct = sum(before_flags)
        selected_correct = sum(after_flags)
        units = len(before_flags)
        cells[cell] = {
            "topics": len(rows),
            "official_units": units,
            "baseline_correct": baseline_correct,
            "selected_correct": selected_correct,
            "delta": selected_correct - baseline_correct,
            "baseline_accuracy": baseline_correct / units,
            "selected_accuracy": selected_correct / units,
            "accuracy_delta": (selected_correct - baseline_correct) / units,
            "corrections": corrections,
            "corruptions": corruptions,
            "selections": selections,
            "observations": observations,
            "mean_observations": observations / len(rows),
            "oracle_fixes": oracle_fixes,
        }
    if indexed_decisions:
        raise ValueError("decision artifact contains unexpected cells or ordinals")

    datasets = _pool_cells(cells, by="dataset")
    backbones = _pool_cells(cells, by="backbone")
    aggregate_counter: Counter[str] = Counter()
    for metrics in cells.values():
        for name in (
            "topics", "official_units", "baseline_correct", "selected_correct",
            "corrections", "corruptions", "selections", "observations",
            "oracle_fixes",
        ):
            aggregate_counter[name] += metrics[name]
    aggregate = dict(aggregate_counter)
    aggregate["delta"] = aggregate["selected_correct"] - aggregate["baseline_correct"]
    aggregate["baseline_accuracy"] = (
        aggregate["baseline_correct"] / aggregate["official_units"]
    )
    aggregate["selected_accuracy"] = (
        aggregate["selected_correct"] / aggregate["official_units"]
    )
    aggregate["accuracy_delta"] = aggregate["delta"] / aggregate["official_units"]
    aggregate["mean_observations"] = aggregate["observations"] / aggregate["topics"]
    audits = {
        "accounting": (
            aggregate["topics"] == expected_topics
            and aggregate["official_units"] == expected_units
        ),
        "input_hashes": True,
        "policy_hash": isinstance(artifact.get("policy_sha256"), str),
        "decision_hash": True,
        "fixed_observations": all(
            value.get("fixed_observations") is True
            for value in inputs.values() if isinstance(value, Mapping)
        ),
    }
    report = {
        "schema_version": 1,
        "artifact_kind": "unified-uncertainty-support-locked-regression",
        "data_scope": "locked_regression_after_label_blind_decisions",
        "policy_sha256": artifact.get("policy_sha256"),
        "decision_sha256": artifact.get("decision_payload_sha256"),
        "cells": cells,
        "datasets": datasets,
        "backbones": backbones,
        "aggregate": aggregate,
        "action_counts": dict(sorted(action_counts.items())),
        "stage3b_comparison": {
            "correct": 201,
            "official_units": 256,
            "accuracy": 201 / 256,
        },
        "audits": audits,
    }
    report["gates"] = evaluate_locked_gates(report)
    return report


def evaluate_locked_gates(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the exact no-regression and accounting contract."""
    failures = []
    cells = report.get("cells")
    aggregate = report.get("aggregate")
    audits = report.get("audits")
    if not isinstance(cells, Mapping) or set(cells) != EXPECTED_CELLS:
        failures.append("locked scope is not the exact eight cells")
        cells = cells if isinstance(cells, Mapping) else {}
    for cell, metrics in sorted(cells.items()):
        if (
            isinstance(metrics, Mapping)
            and metrics.get("selected_correct", -1) < metrics.get("baseline_correct", 0)
        ):
            failures.append(f"{cell} regressed below CVSearch")
    qwen_hr4 = cells.get("qwen/hr_bench_4k")
    if not isinstance(qwen_hr4, Mapping) or qwen_hr4.get("selected_correct", -1) < 33:
        failures.append("qwen/hr_bench_4k below 33/48")

    for pool_kind, index in (("dataset", 1), ("backbone", 0)):
        pooled: dict[str, list[int]] = {}
        for cell, metrics in cells.items():
            if not isinstance(metrics, Mapping):
                continue
            key = cell.split("/")[index]
            values = pooled.setdefault(key, [0, 0])
            values[0] += metrics.get("baseline_correct", 0)
            values[1] += metrics.get("selected_correct", 0)
        for key, (before, after) in sorted(pooled.items()):
            if after < before:
                failures.append(f"{pool_kind} {key} regressed below CVSearch")

    if not isinstance(aggregate, Mapping):
        failures.append("aggregate accounting is missing")
    else:
        if aggregate.get("topics") != 112 or aggregate.get("official_units") != 256:
            failures.append("aggregate scope is not 112 topics/256 units")
        if aggregate.get("baseline_correct") != 195:
            failures.append("CVSearch baseline is not 195/256")
        if aggregate.get("selected_correct", -1) <= 195:
            failures.append("aggregate did not exceed 195/256")
        if aggregate.get("corrections", 0) <= aggregate.get("corruptions", 0):
            failures.append("corrections did not exceed corruptions")
    required_audits = {
        "accounting", "input_hashes", "policy_hash", "decision_hash",
        "fixed_observations",
    }
    if not isinstance(audits, Mapping) or any(
        audits.get(name) is not True for name in required_audits
    ):
        failures.append("one or more provenance/accounting audits failed")
    return {"passed": not failures, "failures": failures}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"JSONL input is not a file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path}:{line_number}")
        value = json.loads(line)
        if type(value) is not dict:
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    _index_rows(rows, str(path))
    return rows


def _load_suite(root: Path) -> dict[str, list[dict[str, Any]]]:
    if not root.is_dir():
        raise ValueError(f"suite root is not a directory: {root}")
    result = {}
    for path in sorted(root.glob("*/*.jsonl")):
        backbone = path.parent.name
        benchmark = path.stem
        cell = f"{backbone}/{benchmark}"
        _split_cell(cell)
        result[cell] = _read_jsonl(path)
    if not result:
        raise ValueError(f"suite root has no backbone/benchmark JSONL files: {root}")
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _load_policy(path: Path) -> UnifiedPolicy:
    payload = _load_json(path)
    if payload.get("payload_sha256") != canonical_payload_hash(payload):
        raise ValueError("frozen unified policy hash mismatch")
    if (
        payload.get("artifact_kind") != "unified-uncertainty-support-policy"
        or payload.get("data_scope") != "opened_development_only"
    ):
        raise ValueError("frozen unified policy scope is invalid")
    policy = UnifiedPolicy.from_dict(payload)
    if payload.get("weights") != list(policy.weights):
        raise ValueError("frozen unified policy weights drifted from profile")
    return policy


def _suite_units(suite: Mapping[str, Sequence[Mapping[str, Any]]]) -> int:
    total = 0
    for cell, rows in suite.items():
        _, benchmark = _split_cell(cell)
        for row in rows:
            total += len(official_correctness(benchmark, row, row.get("output")))
    return total


def _development_records(
    suite: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Any],
) -> list[DevelopmentRecord]:
    records = []
    for cell in sorted(suite):
        backbone, benchmark = _split_cell(cell)
        calibration = calibrations.get(backbone)
        if calibration is None:
            raise ValueError(f"missing development calibration for {backbone}")
        for row in suite[cell]:
            ordinal = row.get("_eg_ordinal")
            records.append(DevelopmentRecord(
                group=source_group(benchmark, ordinal, row.get("input_image")),
                backbone=backbone,
                benchmark=benchmark,
                ordinal=ordinal,
                stage2_row=row,
                split_row=row,
                calibration=calibration,
            ))
    return records


def _development_provenance(
    suite: Mapping[str, Sequence[Mapping[str, Any]]],
    calibrations: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "cells": {
            cell: {
                "rows": len(rows),
                "observations_sha256": _hash_value({
                    row["_eg_ordinal"]: _sanitize_row(row)
                    for row in sorted(rows, key=lambda value: value["_eg_ordinal"])
                }),
            }
            for cell, rows in sorted(suite.items())
        },
        "support_calibrations": {
            backbone: calibration.manifest_sha256
            for backbone, calibration in sorted(calibrations.items())
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only unified uncertainty-support selector replay",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-development")
    freeze.add_argument("--split-root", type=Path, required=True)
    freeze.add_argument("--support-calibration", type=Path, required=True)
    freeze.add_argument("--policy-out", type=Path, required=True)
    freeze.add_argument("--report-out", type=Path, required=True)

    generate = commands.add_parser("generate-decisions")
    generate.add_argument("--stage2-root", type=Path, required=True)
    generate.add_argument("--split-root", type=Path, required=True)
    generate.add_argument("--support-calibration", type=Path, required=True)
    generate.add_argument("--policy", type=Path, required=True)
    generate.add_argument("--decisions-out", type=Path, required=True)

    score = commands.add_parser("score-decisions")
    score.add_argument("--stage2-root", type=Path, required=True)
    score.add_argument("--decisions", type=Path, required=True)
    score.add_argument("--report-out", type=Path, required=True)
    score.add_argument("--expected-topics", type=int, default=112)
    score.add_argument("--expected-units", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze-development":
        suite = _load_suite(args.split_root)
        calibrations = load_selected_calibrations(args.support_calibration)
        records = _development_records(suite, calibrations)
        payload = freeze_policy(
            records,
            provenance=_development_provenance(suite, calibrations),
        )
        _write_json(args.policy_out, payload)
        policy = UnifiedPolicy.from_dict(payload)
        decisions = generate_decisions(suite, suite, calibrations, policy)
        report = score_decisions(
            suite,
            decisions,
            expected_topics=sum(len(rows) for rows in suite.values()),
            expected_units=_suite_units(suite),
        )
        report["artifact_kind"] = (
            "unified-uncertainty-support-opened-development"
        )
        report["data_scope"] = "opened_development_only"
        report["oof_selection"] = copy.deepcopy(payload["oof_metrics"])
        report["selected_profile"] = payload["profile"]
        report["selected_threshold"] = payload["threshold"]
        report.pop("gates", None)
        report["development_gate"] = {
            "passed": True,
            "rule": "source_grouped_oof_every_available_cell_no_harm",
        }
        _write_json(args.report_out, report)
        return 0
    if args.command == "generate-decisions":
        stage2 = _load_suite(args.stage2_root)
        split = _load_suite(args.split_root)
        calibrations = load_selected_calibrations(args.support_calibration)
        policy = _load_policy(args.policy)
        _write_json(
            args.decisions_out,
            generate_decisions(stage2, split, calibrations, policy),
        )
        return 0
    if args.command == "score-decisions":
        report = score_decisions(
            _load_suite(args.stage2_root),
            _load_json(args.decisions),
            expected_topics=args.expected_topics,
            expected_units=args.expected_units,
        )
        _write_json(args.report_out, report)
        return 0 if report["gates"]["passed"] else 2
    raise AssertionError("argparse admitted an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
