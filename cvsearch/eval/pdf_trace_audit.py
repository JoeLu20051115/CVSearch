#!/usr/bin/env python3
"""Strict activity audit for PDF-faithful search traces."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.provenance import canonical_sha256


_BANNED_TRACE_KEYS = frozenset({
    "answer_key", "category", "correct_answer", "ground_truth", "gt_answer",
    "label", "ordinal", "target_object",
})
_RANK_COMPONENTS = frozenset({
    "main", "augmented", "augmented_topk", "complexity", "edge_density",
})
_ACTIONS = ("ZOOM", "SPLIT", "EXPAND", "NEXT", "BACKTRACK")
_TERMINATIONS = frozenset({"CERTIFIED_STOP", "FORCED_RETURN"})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _reject_evaluator_keys(value: Any, path: str = "trace") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            if key.casefold() in _BANNED_TRACE_KEYS:
                raise ValueError(f"{path} contains evaluator-only key {key}")
            _reject_evaluator_keys(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_evaluator_keys(item, f"{path}[{index}]")


def audit_pdf_trace(trace: Mapping[str, Any], *, require_operational: bool = False) -> dict[str, Any]:
    """Validate one trace and return compact, label-free activity counts."""
    payload = _mapping(trace, "trace")
    json.dumps(payload, allow_nan=False)
    _reject_evaluator_keys(payload)
    if payload.get("schema_version") != 1 or payload.get("method") != "pdf_faithful_v1":
        raise ValueError("trace schema or method is invalid")

    plan = _mapping(payload.get("query_plan"), "query_plan")
    qaug = _sequence(plan.get("augmented_queries"), "augmented_queries")
    if len(qaug) < 3 or not all(isinstance(item, str) and item.strip() for item in qaug):
        raise ValueError("query plan must retain at least three augmented queries")

    factory = _mapping(payload.get("candidate_factory"), "candidate_factory")
    collector = _mapping(factory.get("collector"), "candidate collector")
    if factory.get("collector_sha256") != canonical_sha256(collector):
        raise ValueError("candidate collector digest does not match its payload")
    if factory.get("mode") != "cvsearch_tree_only_quick_gate_disabled":
        raise ValueError("candidate factory mode is invalid")

    activity = _mapping(payload.get("module_activity"), "module_activity")
    ranking_activity = _mapping(activity.get("joint_ranking"), "joint ranking activity")
    candidate_count = _integer(ranking_activity.get("candidate_count"), "candidate_count")
    details = _sequence(payload.get("joint_ranking"), "joint_ranking")
    if len(details) != candidate_count:
        raise ValueError("joint ranking details do not cover every candidate")
    identities = set()
    for index, raw_detail in enumerate(details):
        detail = _mapping(raw_detail, f"joint_ranking[{index}]")
        identity = (detail.get("parent_key"), detail.get("canonical_key"))
        if not all(isinstance(item, str) and item for item in identity) or identity in identities:
            raise ValueError("joint ranking candidate identities must be unique and nonempty")
        identities.add(identity)
        if detail.get("top_k_augmented") != 3:
            raise ValueError("full ranking must use the actual Top-3 augmented scores")
        raw_scores = _mapping(detail.get("raw_score"), "raw rank scores")
        if set(raw_scores) != _RANK_COMPONENTS:
            raise ValueError("raw ranking must contain Main, augmented, complexity, and edge")
        top = _sequence(raw_scores["augmented_topk"], "augmented_topk")
        if len(top) != 3:
            raise ValueError("augmented_topk must contain exactly three values")
        for name in ("main", "augmented", "complexity", "edge_density"):
            _finite(raw_scores[name], f"raw ranking {name}")
        for value in top:
            _finite(value, "raw augmented Top-3 value")
        rank = _mapping(detail.get("score"), "fused rank score")
        _finite(rank.get("rank"), "fused rank")
    if ranking_activity.get("top3_candidate_count") != candidate_count:
        raise ValueError("Top-3 activity count does not match ranked candidates")
    if candidate_count and ranking_activity.get("all_candidates_use_true_top3") is not True:
        raise ValueError("Top-3 activity flag is false")

    evaluations = _sequence(payload.get("state_evaluations"), "state_evaluations")
    controller = _mapping(payload.get("controller"), "controller")
    assessment_count = _integer(controller.get("assessment_count"), "assessment_count")
    if len(evaluations) != assessment_count:
        raise ValueError("state evaluations do not align with controller assessments")
    fallback_states = 0
    independent_states = 0
    gap_fallback_states = 0
    root_key = None
    for index, raw_record in enumerate(evaluations):
        record = _mapping(raw_record, f"state_evaluations[{index}]")
        state = _mapping(record.get("state"), "evaluated state")
        path = _sequence(state.get("path_keys"), "evaluated path")
        if not path or not all(isinstance(key, str) and key for key in path):
            raise ValueError("every evaluated state needs a root-to-focus path")
        root_key = path[0] if root_key is None else root_key
        if path[0] != root_key:
            raise ValueError("evaluated paths do not share one root")
        gap = _mapping(record.get("gap"), "gap record")
        scores = _mapping(gap.get("scores"), "gap scores")
        if set(scores) != {"zoom", "split", "expand", "next"}:
            raise ValueError("gap record must contain four independent actions")
        for name, value in scores.items():
            probability = _finite(value, f"gap {name}")
            if not 0.0 <= probability <= 1.0:
                raise ValueError("gap probabilities must be in [0, 1]")
        gap_fallback_states += gap.get("mode") == "analytic_fallback"
        support = _mapping(record.get("support"), "support record")
        independent_states += support.get("independent") is True
        fallback_states += support.get("fallback_used") is True
        if (support.get("independent") is True) == (support.get("fallback_used") is True):
            raise ValueError("support state must be exactly independent or fallback")

    if _mapping(activity.get("uncertainty"), "uncertainty activity").get(
        "state_evaluations"
    ) != len(evaluations):
        raise ValueError("uncertainty activity count does not match state evaluations")
    verifier_activity = _mapping(activity.get("verifier"), "verifier activity")
    if (
        verifier_activity.get("independent_state_count") != independent_states
        or verifier_activity.get("fallback_state_count") != fallback_states
    ):
        raise ValueError("verifier activity counts do not match state records")

    action_counts = Counter()
    for raw_step in _sequence(controller.get("steps"), "controller steps"):
        step = _mapping(raw_step, "controller step")
        action = step.get("action")
        if action not in _ACTIONS:
            raise ValueError("controller step action is invalid")
        if step.get("status") not in {"changed", "no_op"}:
            raise ValueError("controller step status is invalid")
        action_counts[action] += 1
    termination = controller.get("termination")
    if termination not in _TERMINATIONS or activity.get("termination") != termination:
        raise ValueError("controller termination is invalid or inconsistent")
    decision = _mapping(payload.get("final_decision"), "final_decision")
    source = decision.get("source")
    if source not in {"controller", "cvsearch_safety_fallback"}:
        raise ValueError("final decision source is invalid")
    safety_fallback = source == "cvsearch_safety_fallback"
    if activity.get("safety_fallback_used") is not safety_fallback:
        raise ValueError("safety-fallback activity is inconsistent")
    if safety_fallback and termination != "FORCED_RETURN":
        raise ValueError("a certified stop cannot use the CVSearch safety fallback")

    if require_operational:
        if candidate_count == 0:
            raise ValueError("joint ranking was not operational")
        if not evaluations:
            raise ValueError("uncertainty re-answering was not operational")
        if fallback_states or independent_states != len(evaluations):
            raise ValueError("independent verifier was not operational in every state")

    return {
        "ranking_candidates": candidate_count,
        "ranking_groups": _integer(ranking_activity.get("sibling_groups"), "sibling_groups"),
        "native_first_choice_changes": _integer(
            ranking_activity.get("native_first_choice_changes"),
            "native_first_choice_changes",
        ),
        "state_evaluations": len(evaluations),
        "verifier_independent_states": independent_states,
        "verifier_fallback_states": fallback_states,
        "gap_fallback_states": gap_fallback_states,
        "actions": dict(sorted(action_counts.items())),
        "termination": termination,
        "planner_fallback": plan.get("fallback_used") is True,
        "safety_fallback": safety_fallback,
    }


def audit_pdf_traces(traces: Sequence[Mapping[str, Any]], *,
                     require_operational: bool = False) -> dict[str, Any]:
    if isinstance(traces, (str, bytes)) or not isinstance(traces, Sequence) or not traces:
        raise ValueError("traces must be a nonempty sequence")
    reports = [
        audit_pdf_trace(trace, require_operational=require_operational)
        for trace in traces
    ]
    actions = Counter()
    terminations = Counter()
    for report in reports:
        actions.update(report["actions"])
        terminations[report["termination"]] += 1
    return {
        "rows": len(reports),
        "ranking_candidates": sum(item["ranking_candidates"] for item in reports),
        "ranking_groups": sum(item["ranking_groups"] for item in reports),
        "native_first_choice_changes": sum(
            item["native_first_choice_changes"] for item in reports
        ),
        "state_evaluations": sum(item["state_evaluations"] for item in reports),
        "verifier_fallback_states": sum(
            item["verifier_fallback_states"] for item in reports
        ),
        "gap_fallback_states": sum(item["gap_fallback_states"] for item in reports),
        "planner_fallback_rows": sum(item["planner_fallback"] for item in reports),
        "safety_fallback_rows": sum(item["safety_fallback"] for item in reports),
        "actions": dict(sorted(actions.items())),
        "terminations": dict(sorted(terminations.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl")
    parser.add_argument("--require-operational", action="store_true")
    args = parser.parse_args(argv)
    rows = []
    with Path(args.jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(_mapping(row, "JSONL row")["pdf_trace"])
    print(json.dumps(
        audit_pdf_traces(rows, require_operational=args.require_operational),
        sort_keys=True, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
