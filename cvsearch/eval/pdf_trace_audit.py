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

from cvsearch.evidence_gap.pdf_runtime import absolute_location_constraints
from cvsearch.evidence_gap.provenance import canonical_sha256


_BANNED_TRACE_KEYS = frozenset({
    "answer_key", "category", "correct_answer", "ground_truth", "gt_answer",
    "label", "ordinal", "target_object",
})
_RANK_COMPONENTS = frozenset({
    "main", "augmented", "augmented_topk", "complexity", "edge_density",
})
_ACTIONS = ("ZOOM", "SPLIT", "EXPAND", "NEXT", "BACKTRACK")
_TERMINATIONS = frozenset({"CERTIFIED_STOP", "FORCED_RETURN", "BUDGET_FALLBACK"})
_PAIR_MIN_AVG_DELTA = 0.1


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
    evidence_items = [
        _mapping(item, "query plan evidence item")
        for item in _sequence(plan.get("evidence_items"), "evidence_items")
    ]
    expected_verifier_view_policy = (
        "focus_only_target_detail_v1"
        if evidence_items
        and all(item.get("kind") == "target_detail" for item in evidence_items)
        else "overview_plus_detail_v1"
    )

    factory = _mapping(payload.get("candidate_factory"), "candidate_factory")
    collector = _mapping(factory.get("collector"), "candidate collector")
    if factory.get("collector_sha256") != canonical_sha256(collector):
        raise ValueError("candidate collector digest does not match its payload")
    if factory.get("mode") not in {
        "cvsearch_tree_only_quick_gate_disabled",
        "strict_native_p0_lazy_tree_v1",
    }:
        raise ValueError("candidate factory mode is invalid")
    ranking_route = payload.get("ranking_route")
    verification_route = payload.get("verification_route")
    if factory.get("mode") == "strict_native_p0_lazy_tree_v1":
        if ranking_route != {
            "alpha": 0.1,
            "beta": 0.7,
            "visual_lambda": 0.7,
            "protected_head": 3,
        }:
            raise ValueError("strict native P0 ranking route is invalid")
        if verification_route != {
            "mode": "candidate_independent_full_image_v1",
            "min_confidence": 0.9,
            "min_proposal_frequency": 0.4,
        }:
            raise ValueError("strict native P0 verification route is invalid")
    elif ranking_route is not None:
        route = _mapping(ranking_route, "ranking route")
        config_ranking = _mapping(
            _mapping(payload.get("config"), "config").get("ranking"),
            "config ranking",
        )
        if route != {
            "alpha": config_ranking.get("alpha"),
            "beta": config_ranking.get("beta"),
            "visual_lambda": config_ranking.get("visual_lambda"),
            "protected_head": None,
        }:
            raise ValueError("unrouted ranking route is invalid")
        if verification_route is not None and verification_route != {
            "mode": "paired_local_v1",
            "min_confidence": None,
            "min_proposal_frequency": None,
        }:
            raise ValueError("unrouted verification route is invalid")
    all_snapshots = [
        _mapping(item, "candidate snapshot")
        for item in _sequence(collector.get("snapshots"), "candidate snapshots")
    ]
    tree_snapshots = [
        snapshot for snapshot in all_snapshots
        if snapshot.get("event") == "tree_ready"
    ]
    source_identity = _mapping(
        collector.get("source_image_identity"), "collector source image identity",
    )
    source_size = _sequence(source_identity.get("size"), "collector source image size")
    if len(source_size) != 2:
        raise ValueError("collector source image size must contain width and height")
    full_bbox = [0, 0, source_size[0], source_size[1]]
    tree_node_keys = set()
    tree_bbox_by_key = {}
    for snapshot in all_snapshots:
        for raw_node in _sequence(snapshot.get("candidates"), "snapshot candidates"):
            node = _mapping(raw_node, "snapshot candidate")
            key = node.get("canonical_key")
            bbox = _sequence(node.get("bbox_original"), "tree candidate bbox")
            if not isinstance(key, str) or not key or len(bbox) != 4:
                raise ValueError("snapshot candidates need a key and four-value bbox")
            if key in tree_bbox_by_key and tree_bbox_by_key[key] != list(bbox):
                raise ValueError("tree candidate bbox changed across snapshots")
            tree_bbox_by_key[key] = list(bbox)
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
        if record.get("verifier_view_policy") != expected_verifier_view_policy:
            raise ValueError("state verifier view policy does not match the evidence plan")
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
        "independent_full_image", "cvsearch_safety_fallback",
    }:
        raise ValueError("final decision source is invalid")
    if termination == "BUDGET_FALLBACK":
        if (
            source != "cvsearch_safety_fallback"
            or decision.get("reason") != "initial_assessment_exceeds_budget"
            or evaluations or assessment_count != 0 or action_counts
            or controller.get("selected_history_state_id") is not None
            or decision.get("controller_answer_sha256") is not None
            or decision.get("proposal_answer_sha256") is not None
            or decision.get("output_sha256") != factory.get("native_output_sha256")
        ):
            raise ValueError("initial budget fallback is inconsistent")
        budget_fallback = _mapping(
            controller.get("budget_fallback"), "controller budget fallback",
        )
        estimated_calls = _integer(
            budget_fallback.get("estimated_model_calls"),
            "estimated model calls",
        )
        estimated_pixels = _integer(
            budget_fallback.get("estimated_processed_pixels"),
            "estimated processed pixels",
        )
        remaining_calls = _integer(
            budget_fallback.get("remaining_model_calls"),
            "remaining model calls",
        )
        remaining_pixels = _integer(
            budget_fallback.get("remaining_processed_pixels"),
            "remaining processed pixels",
        )
        if estimated_calls <= remaining_calls and estimated_pixels <= remaining_pixels:
            raise ValueError("initial budget fallback did not exceed either budget")
        verifier_activity = _mapping(activity.get("verifier"), "verifier activity")
        if (
            verifier_activity.get("model_calls") != 0
            or verifier_activity.get("paired_reference_calls") != 0
            or activity.get("safety_fallback_used") is not True
            or activity.get("budget_fallback_used") is not True
        ):
            raise ValueError("initial budget fallback activity is inconsistent")
        if require_operational:
            raise ValueError("uncertainty re-answering was not operational")
        return {
            "ranking_candidates": candidate_count,
            "tree_ready_snapshots": len(tree_snapshots),
            "tree_nodes": len(tree_node_keys),
            "max_tree_path_nodes": max_tree_path_nodes,
            "ranking_groups": _integer(
                ranking_activity.get("sibling_groups"), "sibling_groups",
            ),
            "native_first_choice_changes": _integer(
                ranking_activity.get("native_first_choice_changes"),
                "native_first_choice_changes",
            ),
            "state_evaluations": 0,
            "verifier_independent_states": 0,
            "verifier_fallback_states": 0,
            "gap_fallback_states": 0,
            "actions": {},
            "termination": termination,
            "planner_fallback": plan.get("fallback_used") is True,
            "safety_fallback": True,
        }
    if "paired_reference" not in decision:
        raise ValueError("final decision is missing paired reference evidence")
    paired = _mapping(decision["paired_reference"], "paired reference decision")
    if paired.get("verifier_view_policy") != expected_verifier_view_policy:
        raise ValueError("paired verifier view policy does not match the evidence plan")
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
    proposal_state_support = _mapping(
        proposal_record.get("support"), "proposal state support",
    )
    recorded_state_support = paired.get("state_support")
    if recorded_state_support is not None and _mapping(
        recorded_state_support, "paired state support",
    ) != proposal_state_support:
        raise ValueError("paired state support does not match its evaluated state")
    config = _mapping(payload.get("config"), "trace config")
    controller_config = _mapping(config.get("controller"), "controller config")
    expected_state_support_floor = _finite(
        controller_config.get("min_support_min"), "controller minimum support",
    )
    state_support_floor = _finite(
        paired.get("state_support_floor"), "paired state support floor",
    )
    if not math.isclose(
        state_support_floor, expected_state_support_floor,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("paired state support floor does not match the controller")
    expected_state_support_floor_met = (
        proposal_state_support.get("independent") is True
        and proposal_state_support.get("fallback_used") is False
        and _finite(
            proposal_state_support.get("support_min"),
            "proposal state minimum support",
        ) >= state_support_floor
    )
    if paired.get("state_support_floor_met") is not expected_state_support_floor_met:
        raise ValueError("paired state support floor decision is inconsistent")
    if paired.get("attempted") and not expected_state_support_floor_met:
        raise ValueError("paired comparison was attempted below the state support floor")
    proposal_digest = canonical_sha256(
        _mapping(proposal_record.get("answer"), "proposal answer").get("output")
    )
    spatial_focus_keys = set()
    for raw_record in evaluations:
        record = _mapping(raw_record, "state evaluation")
        answer = _mapping(record.get("answer"), "history proposal answer")
        support = _mapping(record.get("support"), "history proposal support")
        if (
            canonical_sha256(answer.get("output")) == proposal_digest
            and answer.get("aggregation_available") is True
            and _finite(
                answer.get("frequency"), "history proposal frequency",
            ) >= 2.0 / 3.0
            and support.get("independent") is True
            and support.get("fallback_used") is False
            and _finite(
                support.get("support_min"), "history proposal minimum support",
            ) >= state_support_floor
        ):
            path = _sequence(
                _mapping(record.get("state"), "history proposal state").get("path_keys"),
                "history proposal path",
            )
            spatial_focus_keys.add(path[-1])
    expected_spatial_count = len(spatial_focus_keys)
    spatial_required = origin == "history_uncertainty_rescue"
    spatial_met = not spatial_required or expected_spatial_count >= 2
    if paired.get("history_spatial_consensus_required") is not spatial_required:
        raise ValueError("paired history spatial-consensus requirement is inconsistent")
    if _integer(
        paired.get("history_spatial_support_count"),
        "paired history spatial-support count",
    ) != expected_spatial_count:
        raise ValueError("paired history spatial-support count is inconsistent")
    if paired.get("history_spatial_consensus_met") is not spatial_met:
        raise ValueError("paired history spatial-consensus decision is inconsistent")
    if paired.get("attempted") and not spatial_met:
        raise ValueError("paired comparison lacks cross-node history consensus")
    geometry = _mapping(paired.get("geometry"), "paired reference geometry")
    constraints = list(_sequence(
        geometry.get("constraints"), "paired geometry constraints",
    ))
    expected_constraints = list(absolute_location_constraints(plan["main_query"]))
    if constraints != expected_constraints:
        raise ValueError("paired geometry constraints do not match the question")
    proposal_state = _mapping(proposal_record.get("state"), "proposal state")
    proposal_path = _sequence(proposal_state.get("path_keys"), "proposal path")
    expected_bbox = tree_bbox_by_key.get(proposal_path[-1])
    raw_bbox = list(_sequence(geometry.get("focus_bbox"), "paired geometry bbox"))
    if expected_bbox is None or raw_bbox != expected_bbox:
        raise ValueError("paired geometry bbox does not match the proposal focus")
    center = tuple(
        _finite(value, "paired geometry center")
        for value in _sequence(
            geometry.get("focus_center_fraction"), "paired geometry center",
        )
    )
    if len(center) != 2:
        raise ValueError("paired geometry center must contain two values")
    x, y, width, height = (_finite(value, "paired geometry bbox") for value in raw_bbox)
    expected_center = (
        (x + width / 2.0) / _finite(source_size[0], "source image width"),
        (y + height / 2.0) / _finite(source_size[1], "source image height"),
    )
    if any(
        not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
        for value, expected in zip(center, expected_center)
    ):
        raise ValueError("paired geometry center does not match the proposal bbox")
    expected_absolute_eligible = all({
        "left": center[0] <= 0.5,
        "right": center[0] >= 0.5,
        "top": center[1] <= 0.5,
        "bottom": center[1] >= 0.5,
    }[name] for name in constraints)
    expected_geometry_eligible = expected_absolute_eligible
    if "absolute_eligible" in geometry:
        if geometry.get("absolute_eligible") is not expected_absolute_eligible:
            raise ValueError("paired absolute geometry eligibility is inconsistent")
        relation_required = any(
            _mapping(item, "query-plan evidence item").get("kind")
            == "relation_context"
            for item in _sequence(
                plan.get("evidence_items"), "query-plan evidence items",
            )
        )
        observation_keys = _sequence(
            proposal_state.get("observation_keys"), "proposal observation keys",
        )
        zoom_prefix = f"{proposal_path[-1]}@zoom"
        zoom_levels = [
            int(item[len(zoom_prefix):])
            for item in observation_keys
            if isinstance(item, str)
            and item.startswith(zoom_prefix)
            and item[len(zoom_prefix):].isdigit()
        ]
        zoom_level = max(zoom_levels, default=0)
        context_count = len(_sequence(
            proposal_state.get("context_keys"), "proposal context keys",
        ))
        relation_enriched = (
            not relation_required or len(proposal_path) == 1
            or zoom_level > 0 or context_count > 0
        )
        detail_required = (
            any(
                _mapping(item, "query-plan evidence item").get("kind")
                == "target_detail"
                for item in _sequence(
                    plan.get("evidence_items"), "query-plan evidence items",
                )
            )
            and not relation_required
        )
        focus_path_depth = len(proposal_path) - 1
        focus_area_fraction = (width * height) / (
            _finite(source_size[0], "source image width")
            * _finite(source_size[1], "source image height")
        )
        detail_resolution_met = (
            not detail_required
            or (
                focus_path_depth > 0
                and (
                    zoom_level > 0
                    or focus_path_depth >= 2
                    or focus_area_fraction <= 0.25
                )
            )
        )
        detail_localized = detail_resolution_met
        if geometry.get("relation_context_required") is not relation_required:
            raise ValueError("paired relation requirement is inconsistent")
        if geometry.get("relation_enriched") is not relation_enriched:
            raise ValueError("paired relation enrichment is inconsistent")
        if geometry.get("detail_localization_required") is not detail_required:
            raise ValueError("paired detail-localization requirement is inconsistent")
        if not math.isclose(
            _finite(
                geometry.get("focus_area_fraction"),
                "paired geometry focus area fraction",
            ),
            focus_area_fraction, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("paired geometry focus area fraction is inconsistent")
        if _integer(
            geometry.get("focus_path_depth"), "paired geometry focus path depth",
        ) != focus_path_depth:
            raise ValueError("paired geometry focus path depth is inconsistent")
        if geometry.get("detail_resolution_met") is not detail_resolution_met:
            raise ValueError("paired detail resolution is inconsistent")
        if geometry.get("detail_localized") is not detail_localized:
            raise ValueError("paired detail localization is inconsistent")
        if _integer(geometry.get("zoom_level"), "paired geometry zoom level") != zoom_level:
            raise ValueError("paired geometry zoom level is inconsistent")
        if _integer(
            geometry.get("context_count"), "paired geometry context count",
        ) != context_count:
            raise ValueError("paired geometry context count is inconsistent")
        expected_geometry_eligible = (
            expected_absolute_eligible and relation_enriched and detail_localized
        )
    if geometry.get("eligible") is not expected_geometry_eligible:
        raise ValueError("paired geometry eligibility is inconsistent")
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
    if decision.get("proposal_answer_sha256") != proposal_digest:
        raise ValueError("proposal answer digest does not match its evaluated state")
    full_image_calls = 0
    full_image = decision.get("independent_full_image")
    if verification_route is not None and _mapping(
        verification_route, "verification route",
    ).get("mode") == "candidate_independent_full_image_v1":
        check = _mapping(full_image, "independent full-image decision")
        required = proposal_digest != factory.get("native_output_sha256")
        for name in ("required", "attempted", "selected"):
            if type(check.get(name)) is not bool:
                raise TypeError(f"independent full-image {name} must be a boolean")
        if check.get("required") is not required:
            raise ValueError("independent full-image requirement is inconsistent")
        if check.get("mode") != "candidate_independent_full_image_v1":
            raise ValueError("independent full-image mode is invalid")
        if not math.isclose(
            _finite(check.get("min_confidence"), "full-image confidence threshold"),
            0.9, rel_tol=0.0, abs_tol=1e-12,
        ) or not math.isclose(
            _finite(
                check.get("min_proposal_frequency"),
                "full-image proposal-frequency threshold",
            ),
            0.4, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("independent full-image thresholds are not frozen")
        answer_record = _mapping(proposal_record.get("answer"), "proposal answer")
        proposal_frequency = _finite(
            check.get("proposal_frequency"), "full-image proposal frequency",
        )
        if not math.isclose(
            proposal_frequency,
            _finite(answer_record.get("frequency"), "proposal answer frequency"),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("independent full-image proposal frequency drifted")
        full_image_calls = _integer(
            check.get("model_calls"), "independent full-image model calls",
        )
        _integer(
            check.get("processed_pixels"),
            "independent full-image processed pixels",
        )
        if check.get("attempted"):
            expected_calls = 3 if isinstance(proposal_answer, int) else 4
            if full_image_calls != expected_calls:
                raise ValueError("independent full-image call accounting is inconsistent")
            failed = check.get("failure_type") is not None
            if failed:
                if (
                    check.get("selected") is not False
                    or check.get("independent_output_sha256") is not None
                    or check.get("independent_confidence") is not None
                    or check.get("aggregation_available") is not None
                    or not isinstance(check.get("failure_message_sha256"), str)
                ):
                    raise ValueError("failed independent full-image check is inconsistent")
            else:
                confidence = _finite(
                    check.get("independent_confidence"),
                    "independent full-image confidence",
                )
                aggregation_available = check.get("aggregation_available")
                if type(aggregation_available) is not bool:
                    raise TypeError(
                        "independent full-image aggregation flag must be boolean"
                    )
                independent_digest = check.get("independent_output_sha256")
                if not isinstance(independent_digest, str):
                    raise TypeError("independent full-image output digest must be a string")
                expected_selected = (
                    aggregation_available
                    and proposal_frequency >= 0.4
                    and confidence >= 0.9
                    and independent_digest == proposal_digest
                )
                if check.get("selected") is not expected_selected:
                    raise ValueError(
                        "independent full-image selection is inconsistent"
                    )
        elif (
            check.get("selected") is not False
            or full_image_calls != 0
            or check.get("independent_output_sha256") is not None
            or check.get("independent_confidence") is not None
            or check.get("aggregation_available") is not None
            or check.get("failure_type") is not None
            or check.get("failure_message_sha256") is not None
        ):
            raise ValueError("unattempted independent full-image check contains work")
        if check.get("selected") is not (source == "independent_full_image"):
            raise ValueError("independent full-image selection and source disagree")
    elif source == "independent_full_image":
        raise ValueError("unrouted run selected independent full-image output")
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
        comparison_mode = paired.get("comparison_mode")
        if comparison_mode is None:
            if proposed != proposal_state_support:
                raise ValueError("legacy paired proposed support does not match its state")
        elif comparison_mode not in {
            "conditional_option_loss", "normalized_yes_no_fallback",
            "worst_case_3x_conditional_option_loss",
            "worst_case_3x_normalized_yes_no_fallback",
        }:
            raise ValueError("paired comparison mode is invalid")
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
        if comparison_mode is not None:
            if recorded_state_support is None:
                raise ValueError("contrastive pairing is missing its state support")
            if any(
                not math.isclose(
                    proposed_value + reference_value, 1.0,
                    rel_tol=0.0, abs_tol=1e-12,
                )
                for proposed_value, reference_value in zip(
                    proposed_values, reference_values,
                )
            ):
                raise ValueError("contrastive paired probabilities are not normalized")
            worst_case = comparison_mode.startswith("worst_case_3x_")
            if worst_case:
                paraphrase_ids = tuple(_sequence(
                    paired.get("paraphrase_ids"), "paired paraphrase ids",
                ))
                if paraphrase_ids != (
                    "question", "requirement_conditioned", "coarse_to_fine",
                ):
                    raise ValueError("paired paraphrase ids do not match the frozen set")
                proposed_rows = tuple(
                    tuple(
                        _finite(value, "paired proposed paraphrase probability")
                        for value in _sequence(row, "paired proposed paraphrase row")
                    )
                    for row in _sequence(
                        paired.get("proposed_by_requirement"),
                        "paired proposed paraphrase matrix",
                    )
                )
                reference_rows = tuple(
                    tuple(
                        _finite(value, "paired reference paraphrase probability")
                        for value in _sequence(row, "paired reference paraphrase row")
                    )
                    for row in _sequence(
                        paired.get("reference_by_requirement"),
                        "paired reference paraphrase matrix",
                    )
                )
                if (
                    len(proposed_rows) != len(proposed_values)
                    or len(reference_rows) != len(reference_values)
                    or any(len(row) != 3 for row in proposed_rows + reference_rows)
                ):
                    raise ValueError("paired paraphrase matrices are not aligned")
                if any(
                    not math.isclose(proposed + reference, 1.0,
                                     rel_tol=0.0, abs_tol=1e-12)
                    for proposed_row, reference_row in zip(
                        proposed_rows, reference_rows,
                    )
                    for proposed, reference in zip(proposed_row, reference_row)
                ):
                    raise ValueError("paired paraphrase probabilities are not normalized")
                if any(
                    proposed != min(row)
                    for proposed, row in zip(proposed_values, proposed_rows)
                ) or any(
                    reference != max(row)
                    for reference, row in zip(reference_values, reference_rows)
                ):
                    raise ValueError("paired supports are not worst-case paraphrase scores")
            base_calls = (
                3 if comparison_mode.endswith("conditional_option_loss") else 2
            )
            expected_calls = base_calls * len(proposed_values) * (3 if worst_case else 1)
            if extra_calls != expected_calls:
                raise ValueError("contrastive paired call accounting is inconsistent")
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
        min_avg_delta = _finite(
            paired.get("min_avg_delta"), "paired minimum average delta",
        )
        min_requirement_delta = _finite(
            paired.get("min_requirement_delta"),
            "paired minimum requirement delta",
        )
        if not math.isclose(
            min_avg_delta, _PAIR_MIN_AVG_DELTA, rel_tol=0.0, abs_tol=1e-12,
        ) or not math.isclose(
            min_requirement_delta, 0.0, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("paired comparison thresholds are not frozen")
        answer_record = _mapping(proposal_record.get("answer"), "proposal answer")
        stable = (
            answer_record.get("aggregation_available") is True
            and _finite(
                answer_record.get("frequency"), "proposal answer frequency",
            ) >= 2.0 / 3.0
        )
        expected_selected = (
            proposed.get("independent") is True
            and reference.get("independent") is True
            and proposed.get("fallback_used") is False
            and reference.get("fallback_used") is False
            and expected_wins == len(deltas)
            and expected_avg_delta > min_avg_delta
            and expected_min_delta >= min_requirement_delta
            and stable
        )
        if paired.get("selected") is not expected_selected:
            raise ValueError("paired selection does not match its frozen thresholds")
    else:
        legacy_unattempted_proposed = (
            "comparison_mode" not in paired
            and paired.get("proposed") == proposal_state_support
        )
        if (
            extra_calls != 0 or paired.get("reference") is not None
            or (
                paired.get("proposed") is not None
                and not legacy_unattempted_proposed
            )
            or paired.get("comparison_mode") is not None
            or paired.get("paraphrase_ids") is not None
            or paired.get("proposed_by_requirement") is not None
            or paired.get("reference_by_requirement") is not None
        ):
            raise ValueError("unattempted paired reference contains verifier work")
    safety_fallback = source == "cvsearch_safety_fallback"
    if activity.get("safety_fallback_used") is not safety_fallback:
        raise ValueError("safety-fallback activity is inconsistent")
    if (
        safety_fallback
        and termination != "FORCED_RETURN"
        and not (
            (paired["required"] and not paired["selected"])
            or (
                isinstance(full_image, Mapping)
                and full_image.get("required") is True
                and full_image.get("selected") is False
            )
        )
    ):
        raise ValueError("a certified stop can fall back only after a paired veto")
    verifier_calls = _integer(verifier_activity.get("model_calls"), "verifier model calls")
    state_calls = sum(
        _integer(_mapping(record.get("support"), "support record").get("model_calls"),
                 "support model calls")
        for record in evaluations
    )
    if verifier_activity.get("candidate_independent_full_image_calls", 0) != full_image_calls:
        raise ValueError("independent full-image activity accounting is inconsistent")
    if verifier_calls != state_calls + extra_calls + full_image_calls:
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
