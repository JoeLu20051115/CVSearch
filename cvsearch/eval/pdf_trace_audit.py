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
    tree_snapshots = [
        _mapping(item, "candidate snapshot")
        for item in _sequence(collector.get("snapshots"), "candidate snapshots")
        if _mapping(item, "candidate snapshot").get("event") == "tree_ready"
    ]
    source_identity = _mapping(
        collector.get("source_image_identity"), "collector source image identity",
    )
    source_size = _sequence(source_identity.get("size"), "collector source image size")
    if len(source_size) != 2:
        raise ValueError("collector source image size must contain width and height")
    full_bbox = [0, 0, source_size[0], source_size[1]]
    tree_node_keys = set()
    max_tree_path_nodes = 0
    for snapshot in tree_snapshots:
        raw_nodes = _sequence(snapshot.get("candidates"), "tree candidates")
        nodes = {}
        for raw_node in raw_nodes:
            node = _mapping(raw_node, "tree candidate")
            key = node.get("canonical_key")
            if not isinstance(key, str) or not key or key in nodes:
                raise ValueError("tree candidate keys must be unique and nonempty")
            nodes[key] = node
            tree_node_keys.add(key)
        roots = [
            (key, node) for key, node in nodes.items()
            if node.get("parent_key") is None
            and node.get("depth") == 0
            and node.get("bbox_original") == full_bbox
        ]
        for root_key, _ in roots:
            queue = [(root_key, 1)]
            seen = set()
            while queue:
                key, path_nodes = queue.pop(0)
                if key in seen:
                    raise ValueError("root-to-leaf tree contains a cycle")
                seen.add(key)
                max_tree_path_nodes = max(max_tree_path_nodes, path_nodes)
                children = _sequence(nodes[key].get("child_keys"), "tree child keys")
                queue.extend(
                    (child, path_nodes + 1) for child in children
                    if child in nodes
                )

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
    if source not in {
        "controller", "controller_paired_reference", "history_paired_reference",
        "cvsearch_safety_fallback",
    }:
        raise ValueError("final decision source is invalid")
    if "paired_reference" not in decision:
        raise ValueError("final decision is missing paired reference evidence")
    paired = _mapping(decision["paired_reference"], "paired reference decision")
    for name in ("required", "attempted", "selected"):
        if type(paired.get(name)) is not bool:
            raise TypeError(f"paired reference {name} must be a boolean")
    paired_state_id = _integer(paired.get("state_id"), "paired reference state id")
    origin = paired.get("origin")
    if origin not in {"controller", "history_uncertainty_rescue"}:
        raise ValueError("paired reference origin is invalid")
    proposal_records = [
        _mapping(record, "state evaluation") for record in evaluations
        if _mapping(record, "state evaluation").get("state", {}).get("state_id")
        == paired_state_id
    ]
    if len(proposal_records) != 1:
        raise ValueError("paired reference state must identify one evaluated state")
    proposal_record = proposal_records[0]
    controller_state_id = _integer(
        controller.get("selected_history_state_id"), "controller selected state id",
    )
    if (origin == "controller") != (paired_state_id == controller_state_id):
        raise ValueError("paired reference origin and state id disagree")
    extra_calls = _integer(
        paired.get("extra_model_calls"), "paired reference model calls",
    )
    _integer(
        paired.get("extra_processed_pixels"), "paired reference processed pixels",
    )
    if paired["selected"] != source.endswith("_paired_reference"):
        raise ValueError("paired reference selection and final source disagree")
    if source == "history_paired_reference" and origin != "history_uncertainty_rescue":
        raise ValueError("history paired selection lacks a history proposal")
    proposal_answer = _mapping(proposal_record.get("answer"), "proposal answer").get("output")
    proposal_digest = canonical_sha256(proposal_answer)
    if decision.get("proposal_answer_sha256") != proposal_digest:
        raise ValueError("proposal answer digest does not match its evaluated state")
    controller_records = [
        _mapping(record, "state evaluation") for record in evaluations
        if _mapping(record, "state evaluation").get("state", {}).get("state_id")
        == controller_state_id
    ]
    if len(controller_records) != 1 or decision.get("controller_answer_sha256") != canonical_sha256(
        _mapping(controller_records[0].get("answer"), "controller answer").get("output")
    ):
        raise ValueError("controller answer digest does not match its selected state")
    expected_output_digest = (
        factory.get("native_output_sha256")
        if source == "cvsearch_safety_fallback" else proposal_digest
    )
    if decision.get("output_sha256") != expected_output_digest:
        raise ValueError("final output digest does not match its decision source")
    if paired["attempted"]:
        if not paired["required"] or extra_calls == 0:
            raise ValueError("paired reference attempt is inconsistent")
        proposed = _mapping(paired.get("proposed"), "paired proposed support")
        reference = _mapping(paired.get("reference"), "paired reference support")
        if proposed != _mapping(proposal_record.get("support"), "proposal support"):
            raise ValueError("paired proposed support does not match its evaluated state")
        if (
            proposed.get("requirement_ids") != reference.get("requirement_ids")
            or proposed.get("checkpoint_sha256") != reference.get("checkpoint_sha256")
        ):
            raise ValueError("paired supports do not share requirements and checkpoint")
        proposed_values = tuple(
            _finite(value, "paired proposed requirement support")
            for value in _sequence(
                proposed.get("per_requirement"), "paired proposed requirements",
            )
        )
        reference_values = tuple(
            _finite(value, "paired reference requirement support")
            for value in _sequence(
                reference.get("per_requirement"), "paired reference requirements",
            )
        )
        if not proposed_values or len(proposed_values) != len(reference_values):
            raise ValueError("paired support vectors must be nonempty and aligned")
        deltas = tuple(
            proposed_value - reference_value
            for proposed_value, reference_value in zip(
                proposed_values, reference_values,
            )
        )
        expected_avg_delta = (
            math.fsum(proposed_values) - math.fsum(reference_values)
        ) / len(deltas)
        expected_min_delta = min(deltas)
        expected_wins = sum(delta > 0.0 for delta in deltas)
        if not math.isclose(
            _finite(paired.get("avg_delta"), "paired average delta"),
            expected_avg_delta, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("paired average delta does not match support vectors")
        if not math.isclose(
            _finite(paired.get("min_delta"), "paired minimum delta"),
            expected_min_delta, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("paired minimum delta does not match support vectors")
        if _integer(
            paired.get("requirement_wins"), "paired requirement wins",
        ) != expected_wins:
            raise ValueError("paired requirement wins do not match support vectors")
        if _integer(
            paired.get("requirement_count"), "paired requirement count",
        ) != len(deltas):
            raise ValueError("paired requirement count does not match support vectors")
    elif extra_calls != 0 or paired.get("reference") is not None:
        raise ValueError("unattempted paired reference contains verifier work")
    safety_fallback = source == "cvsearch_safety_fallback"
    if activity.get("safety_fallback_used") is not safety_fallback:
        raise ValueError("safety-fallback activity is inconsistent")
    if (
        safety_fallback
        and termination != "FORCED_RETURN"
        and not (paired["required"] and not paired["selected"])
    ):
        raise ValueError("a certified stop can fall back only after a paired veto")
    verifier_calls = _integer(verifier_activity.get("model_calls"), "verifier model calls")
    state_calls = sum(
        _integer(_mapping(record.get("support"), "support record").get("model_calls"),
                 "support model calls")
        for record in evaluations
    )
    if verifier_calls != state_calls + extra_calls:
        raise ValueError("verifier model-call accounting is inconsistent")

    if require_operational:
        if max_tree_path_nodes < 2:
            raise ValueError("candidate factory did not materialize a root-to-leaf tree")
        if candidate_count == 0:
            raise ValueError("joint ranking was not operational")
        if not evaluations:
            raise ValueError("uncertainty re-answering was not operational")
        if fallback_states or independent_states != len(evaluations):
            raise ValueError("independent verifier was not operational in every state")

    return {
        "ranking_candidates": candidate_count,
        "tree_ready_snapshots": len(tree_snapshots),
        "tree_nodes": len(tree_node_keys),
        "max_tree_path_nodes": max_tree_path_nodes,
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
        "tree_ready_snapshots": sum(item["tree_ready_snapshots"] for item in reports),
        "tree_nodes": sum(item["tree_nodes"] for item in reports),
        "minimum_max_tree_path_nodes": min(
            item["max_tree_path_nodes"] for item in reports
        ),
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
