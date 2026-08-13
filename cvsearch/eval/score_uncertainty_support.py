#!/usr/bin/env python3
"""Freeze, generate, and score CPU-only unified-selector replays."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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
from cvsearch.eval.replay_adaptive_search import _rank_digest
from cvsearch.evidence_gap.types import (
    AnswerRecord,
    BudgetLedger,
    P0Anchor,
    SplitBranchObservation,
    SplitProbeObservation,
    SplitSearchAudit,
    SplitViewObservation,
)

from .freeze_uncertainty_support import (
    BENCHMARKS,
    DevelopmentRecord,
    canonical_payload_hash,
    freeze_policy,
    source_group,
)
from .replay_uncertainty_support import (
    UnifiedPolicy,
    fail_closed_uncertainty_support,
    replay_uncertainty_support,
    sanitize_replay_row,
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
LOCKED_CELL_SCOPE = {
    f"{backbone}/{benchmark}": (
        (12, 48) if benchmark.startswith("hr_bench_")
        else (12, 12) if benchmark == "treebench"
        else (20, 20)
    )
    for backbone in EXPECTED_BACKBONES
    for benchmark in EXPECTED_BENCHMARKS
}


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_dict(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} does not use its exact frozen schema")
    return value


def _tuple(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON list")
    return tuple(value)


def _answer_record(value: Any) -> AnswerRecord:
    payload = _exact_dict(value, {
        "output", "canonical_answer", "raw_outputs", "groups", "frequency",
        "margin", "confidence", "uncertainty", "losses", "selected_from",
        "aggregation_available", "aggregation_reason",
    }, "split P0 stability")
    return AnswerRecord(
        output=copy.deepcopy(payload["output"]),
        canonical_answer=copy.deepcopy(payload["canonical_answer"]),
        raw_outputs=_tuple(payload["raw_outputs"], "P0 raw outputs"),
        groups=copy.deepcopy(payload["groups"]),
        frequency=payload["frequency"], margin=payload["margin"],
        confidence=payload["confidence"], uncertainty=payload["uncertainty"],
        losses=_tuple(payload["losses"], "P0 losses"),
        selected_from=payload["selected_from"],
        aggregation_available=payload["aggregation_available"],
        aggregation_reason=payload["aggregation_reason"],
    )


def _ledger(value: Any) -> BudgetLedger:
    payload = _exact_dict(value, {
        "max_mllm_calls", "max_processed_pixels", "mllm_calls",
        "processed_pixels",
    }, "split budget ledger")
    return BudgetLedger(**payload)


def _split_probe(value: Any) -> SplitProbeObservation:
    payload = _exact_dict(value, {
        "patch_path", "target_box_xyxy", "source_size", "render_sha256",
        "raw_support", "ranking_score", "mllm_calls", "processed_pixels",
    }, "split screening probe")
    return SplitProbeObservation(
        patch_path=_tuple(payload["patch_path"], "probe path"),
        target_box_xyxy=_tuple(payload["target_box_xyxy"], "probe box"),
        source_size=_tuple(payload["source_size"], "probe source size"),
        render_sha256=payload["render_sha256"],
        raw_support=payload["raw_support"],
        ranking_score=payload["ranking_score"],
        mllm_calls=payload["mllm_calls"],
        processed_pixels=payload["processed_pixels"],
    )


def _split_view(value: Any) -> SplitViewObservation:
    payload = _exact_dict(value, {
        "role", "patch_path", "target_box_xyxy", "crop_xyxy", "source_size",
        "render_sha256", "raw_support", "answer", "mllm_calls",
        "processed_pixels",
    }, "split view")
    return SplitViewObservation(
        role=payload["role"],
        patch_path=_tuple(payload["patch_path"], "view path"),
        target_box_xyxy=_tuple(payload["target_box_xyxy"], "view target box"),
        crop_xyxy=_tuple(payload["crop_xyxy"], "view crop"),
        source_size=_tuple(payload["source_size"], "view source size"),
        render_sha256=payload["render_sha256"],
        raw_support=payload["raw_support"], answer=copy.deepcopy(payload["answer"]),
        mllm_calls=payload["mllm_calls"],
        processed_pixels=payload["processed_pixels"],
    )


def _ranked_siblings(value: Any) -> tuple[tuple[Any, ...], ...]:
    if type(value) is not list:
        raise TypeError("ranked siblings must be a JSON list")
    result = []
    for sibling in value:
        payload = _exact_dict(
            sibling, {"path", "box", "score"}, "ranked split sibling",
        )
        result.append((
            _tuple(payload["path"], "ranked sibling path"),
            _tuple(payload["box"], "ranked sibling box"),
            payload["score"],
        ))
    return tuple(result)


def _split_branch(value: Any) -> SplitBranchObservation:
    if type(value) is not dict:
        raise TypeError("split branch must be an exact dictionary")
    fields = {
        "visit_index", "selected_sibling_rank", "observed_path",
        "ranked_siblings", "tight_view", "context_view", "backtracked",
    }
    if "medium_view" in value:
        fields.add("medium_view")
    payload = _exact_dict(value, fields, "split branch")
    ranked = _ranked_siblings(payload["ranked_siblings"])
    branch = SplitBranchObservation(
        visit_index=payload["visit_index"],
        selected_sibling_rank=payload["selected_sibling_rank"],
        ranked_sibling_paths=tuple(item[0] for item in ranked),
        ranked_sibling_boxes=tuple(item[1] for item in ranked),
        ranked_sibling_scores=tuple(item[2] for item in ranked),
        tight_view=_split_view(payload["tight_view"]),
        medium_view=(
            _split_view(payload["medium_view"])
            if "medium_view" in payload else None
        ),
        context_view=_split_view(payload["context_view"]),
        backtracked=payload["backtracked"],
    )
    if tuple(payload["observed_path"]) != branch.observed_path:
        raise ValueError("split branch observed path is not derived from its tight view")
    return branch


def _p0_anchor(value: Any) -> P0Anchor:
    payload = _exact_dict(value, {
        "emitted_answer", "cvsearch_raw", "producing_phase", "node_keys",
        "support_view",
    }, "split P0 anchor")
    if payload["support_view"] is not None:
        raise ValueError("fixed replay requires a global P0 without local descriptors")
    return P0Anchor(
        emitted_answer=copy.deepcopy(payload["emitted_answer"]),
        cvsearch_raw=copy.deepcopy(payload["cvsearch_raw"]),
        producing_phase=payload["producing_phase"],
        node_keys=_tuple(payload["node_keys"], "P0 node keys"),
        support_view=None,
    )


def _validate_stage3b_visit_order(
    roots: tuple[tuple[int, ...], ...],
    probes: tuple[SplitProbeObservation, ...],
    branches: tuple[SplitBranchObservation, ...],
) -> None:
    ranked_by_root = {}
    for branch in branches:
        root = branch.observed_path[:1]
        material = tuple(zip(
            branch.ranked_sibling_paths,
            branch.ranked_sibling_boxes,
            branch.ranked_sibling_scores,
        ))
        if root in ranked_by_root and ranked_by_root[root] != material:
            raise ValueError("split branches disagree on their frozen sibling ranking")
        ranked_by_root[root] = material
    if set(ranked_by_root) != set(roots):
        raise ValueError("split branches do not bind all four ranked roots")
    expected_probes = tuple(
        item for root in roots for item in ranked_by_root[root]
    )
    observed_probes = tuple(
        (probe.patch_path, probe.target_box_xyxy, probe.ranking_score)
        for probe in probes
    )
    if observed_probes != expected_probes:
        raise ValueError("screening probes drifted from root and sibling rank order")
    first_eight = probes[:8]
    selected = [first_eight[0], first_eight[4]]
    selected_paths = {probe.patch_path for probe in selected}
    for probe in sorted(
        first_eight,
        key=lambda item: (
            -item.raw_support, -item.ranking_score, item.patch_path,
        ),
    ):
        if probe.patch_path not in selected_paths:
            selected.append(probe)
            selected_paths.add(probe.patch_path)
        if len(selected) == 4:
            break
    for offset in (8, 12):
        selected.append(min(
            probes[offset:offset + 4],
            key=lambda item: (
                -item.raw_support, -item.ranking_score, item.patch_path,
            ),
        ))
    if tuple(branch.observed_path for branch in branches) != tuple(
        probe.patch_path for probe in selected
    ):
        raise ValueError("split branch visitation drifted from frozen probe ordering")


def _validated_split_audit(value: Any) -> SplitSearchAudit:
    payload = _exact_dict(value, {
        "p0_anchor", "p0_stability", "root_ranked_siblings", "branches",
        "screening_probes", "rank_sha256", "query_sha256", "render_policy",
        "max_depth", "max_observed_branches", "max_screening_probes",
        "ledger_before", "ledger_after", "no_op_reason",
    }, "split search audit")
    roots = _ranked_siblings(payload["root_ranked_siblings"])
    if any(
        isinstance(item[2], bool) or not isinstance(item[2], (int, float))
        or not math.isfinite(float(item[2])) or not 0.0 <= item[2] <= 1.0
        for item in roots
    ):
        raise ValueError("root ranking scores must be finite unit values")
    branches = tuple(_split_branch(item) for item in payload["branches"])
    probes = tuple(_split_probe(item) for item in payload["screening_probes"])
    audit = SplitSearchAudit(
        p0_anchor=_p0_anchor(payload["p0_anchor"]),
        p0_stability=_answer_record(payload["p0_stability"]),
        branches=branches,
        root_ranked_paths=tuple(item[0] for item in roots),
        root_ranked_boxes=tuple(item[1] for item in roots),
        root_ranked_scores=tuple(item[2] for item in roots),
        rank_sha256=payload["rank_sha256"],
        query_sha256=payload["query_sha256"],
        render_policy=payload["render_policy"],
        ledger_before=_ledger(payload["ledger_before"]),
        ledger_after=_ledger(payload["ledger_after"]),
        screening_probes=probes,
        no_op_reason=payload["no_op_reason"],
    )
    if audit.to_dict() != payload:
        raise ValueError("split audit differs from its reconstructed frozen form")
    if branches:
        _validate_stage3b_visit_order(
            audit.root_ranked_paths, probes, branches,
        )
    return audit


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


def _validate_fixed_observation_contract(row: Mapping[str, Any]) -> SplitSearchAudit:
    audit = _split_audit(row)
    if not isinstance(audit, Mapping) or not (
        audit.get("max_depth") == 2
        and audit.get("max_observed_branches") == 6
        and audit.get("max_screening_probes") == 16
        and audit.get("render_policy")
        == "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
    ):
        raise ValueError("fixed Stage-3b observation header is invalid")
    validated = _validated_split_audit(audit)
    phase1_digest = _rank_digest(row)
    if phase1_digest is None or validated.rank_sha256 != phase1_digest:
        raise ValueError("fixed Stage-3b rank provenance drifted")
    trace = row.get("method_trace")
    query_plan = trace.get("query_plan") if isinstance(trace, Mapping) else None
    if not isinstance(query_plan, Mapping) or (
        validated.query_sha256 != _hash_value(query_plan)
    ):
        raise ValueError("fixed Stage-3b query provenance drifted")
    if validated.no_op_reason == "split_invalid_evidence_requirements":
        if validated.screening_probes or validated.branches:
            raise ValueError("explicit SPLIT no-op retained observations")
        return validated
    if not (
        validated.no_op_reason is None
        and len(validated.screening_probes) == 16
        and len(validated.root_ranked_paths) == 4
        and len(validated.branches) == 6
    ):
        raise ValueError("fixed Stage-3b observation budget is incomplete")
    return validated


def _fixed_observation_contract(row: Mapping[str, Any]) -> bool:
    try:
        _validate_fixed_observation_contract(row)
    except (KeyError, TypeError, ValueError):
        return False
    return True


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
            ordinal: sanitize_replay_row(stage2[ordinal]) for ordinal in sorted(stage2)
        }
        sanitized_split = {
            ordinal: sanitize_replay_row(split[ordinal]) for ordinal in sorted(split)
        }
        observation_errors = {}
        for ordinal, row in sanitized_split.items():
            try:
                _validate_fixed_observation_contract(row)
            except (KeyError, TypeError, ValueError) as error:
                observation_errors[ordinal] = str(error)
        inputs[cell] = {
            "rows": len(stage2),
            "stage2_observations_sha256": _hash_value(sanitized_stage2),
            "split_observations_sha256": _hash_value(sanitized_split),
            "fixed_observations": all(
                ordinal not in observation_errors for ordinal in sanitized_split
            ),
        }
        for ordinal in sorted(stage2):
            decision = (
                fail_closed_uncertainty_support(
                    sanitized_stage2[ordinal], sanitized_split[ordinal],
                    calibration, policy, observation_errors[ordinal],
                )
                if ordinal in observation_errors else
                replay_uncertainty_support(
                    sanitized_stage2[ordinal], sanitized_split[ordinal],
                    calibration, policy,
                )
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
                    for output in (
                        () if ordinal in observation_errors
                        else candidate_outputs(sanitized_split[ordinal])
                    )
                ],
            }
            if "failure_detail" in decision:
                record["failure_detail"] = decision["failure_detail"]
            decisions.append(record)
    payload = {
        "schema_version": 2,
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
    if (
        artifact.get("schema_version") != 2
        or artifact.get("artifact_kind")
        != "unified-uncertainty-support-decisions"
        or artifact.get("data_scope") != "label_blind_locked_replay"
    ):
        raise ValueError("decision artifact schema or scope is invalid")
    if artifact.get("decision_payload_sha256") != _artifact_hash(
        artifact, "decision_payload_sha256",
    ):
        raise ValueError("decision artifact hash mismatch")
    raw_decisions = artifact.get("decisions")
    inputs = artifact.get("inputs")
    support_calibrations = artifact.get("support_calibrations")
    policy_sha256 = artifact.get("policy_sha256")
    if (
        not isinstance(raw_decisions, list)
        or type(inputs) is not dict
        or type(support_calibrations) is not dict
        or not _is_sha256(policy_sha256)
    ):
        raise ValueError("decision artifact is incomplete")
    if set(inputs) != set(labeled_stage2_by_cell):
        raise ValueError("decision artifact input cells differ from scoring cells")
    expected_backbones = {
        _split_cell(cell)[0] for cell in labeled_stage2_by_cell
    }
    if (
        set(support_calibrations) != expected_backbones
        or not all(_is_sha256(value) for value in support_calibrations.values())
    ):
        raise ValueError("decision support-calibration bindings are invalid")
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
            ordinal: sanitize_replay_row(rows[ordinal]) for ordinal in sorted(rows)
        }
        binding = inputs.get(cell)
        if (
            type(binding) is not dict
            or set(binding) != {
                "rows", "stage2_observations_sha256",
                "split_observations_sha256", "fixed_observations",
            }
            or binding.get("rows") != len(rows)
            or binding.get("stage2_observations_sha256") != _hash_value(sanitized)
            or not _is_sha256(binding.get("split_observations_sha256"))
            or type(binding.get("fixed_observations")) is not bool
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
        "support_calibrations": True,
        "policy_hash": _is_sha256(policy_sha256),
        "decision_hash": _is_sha256(artifact.get("decision_payload_sha256")),
        "fixed_observations": all(
            value["fixed_observations"] is True for value in inputs.values()
        ),
    }
    input_bindings = copy.deepcopy(inputs)
    calibration_bindings = copy.deepcopy(support_calibrations)
    report = {
        "schema_version": 2,
        "artifact_kind": "unified-uncertainty-support-locked-regression",
        "data_scope": "locked_regression_after_label_blind_decisions",
        "policy_sha256": policy_sha256,
        "decision_sha256": artifact.get("decision_payload_sha256"),
        "input_bindings": input_bindings,
        "input_bindings_sha256": _hash_value(input_bindings),
        "support_calibrations": calibration_bindings,
        "support_calibrations_sha256": _hash_value(calibration_bindings),
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
        expected_scope = LOCKED_CELL_SCOPE.get(cell)
        if (
            not isinstance(metrics, Mapping)
            or expected_scope is None
            or (metrics.get("topics"), metrics.get("official_units"))
            != expected_scope
        ):
            expected_topics, expected_units = expected_scope or ("?", "?")
            failures.append(
                f"{cell} scope is not {expected_topics} topics/"
                f"{expected_units} units"
            )
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
    input_bindings = report.get("input_bindings")
    support_calibrations = report.get("support_calibrations")
    provenance_valid = (
        type(input_bindings) is dict
        and set(input_bindings) == EXPECTED_CELLS
        and report.get("input_bindings_sha256") == _hash_value(input_bindings)
        and all(
            type(binding) is dict
            and set(binding) == {
                "rows", "stage2_observations_sha256",
                "split_observations_sha256", "fixed_observations",
            }
            and binding["rows"] == LOCKED_CELL_SCOPE[cell][0]
            and _is_sha256(binding["stage2_observations_sha256"])
            and _is_sha256(binding["split_observations_sha256"])
            and binding["fixed_observations"] is True
            for cell, binding in input_bindings.items()
        )
        and type(support_calibrations) is dict
        and set(support_calibrations) == set(EXPECTED_BACKBONES)
        and all(_is_sha256(value) for value in support_calibrations.values())
        and report.get("support_calibrations_sha256")
        == _hash_value(support_calibrations)
        and _is_sha256(report.get("policy_sha256"))
        and _is_sha256(report.get("decision_sha256"))
    )
    if not provenance_valid:
        failures.append("locked input/calibration hash manifest is invalid")
    required_audits = {
        "accounting", "input_hashes", "policy_hash", "decision_hash",
        "fixed_observations", "support_calibrations",
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
        payload.get("schema_version") != 2
        or not _is_sha256(payload.get("source_group_assignments_sha256"))
        or not _is_sha256(payload.get("oof_folds_sha256"))
        or payload.get("artifact_kind")
        != "unified-uncertainty-support-policy"
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
            _validate_fixed_observation_contract(sanitize_replay_row(row))
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
                    row["_eg_ordinal"]: sanitize_replay_row(row)
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
        report["source_group_assignments_sha256"] = payload[
            "source_group_assignments_sha256"
        ]
        report["oof_folds_sha256"] = payload["oof_folds_sha256"]
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
