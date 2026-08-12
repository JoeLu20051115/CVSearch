"""Label-blind Stage 3 replay over candidate-only split observations."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from cvsearch.evidence_gap.split_search import confirm_split_branch, mann_kendall_s
from cvsearch.eval.replay_adaptive_search import (
    FrozenCalibration,
    FrozenSelectedCalibration,
    _answer_record,
    _canonical_json,
    _p0_canonical_answer,
    _rank_digest,
    replay_adaptive_search,
)


_POLICY_FIELDS = frozenset({
    "minimum_final_support", "minimum_support_gain", "maximum_support_drop",
    "minimum_conflict_margin", "minimum_uncontested_support",
    "minimum_consensus_raw_support",
})


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _policy(value: Mapping[str, Any]) -> dict[str, float]:
    if type(value) is not dict or set(value) != _POLICY_FIELDS:
        raise ValueError("split replay policy must use the exact answer-free schema")
    return {
        "minimum_final_support": _unit(
            value["minimum_final_support"], "minimum_final_support",
        ),
        "minimum_support_gain": _unit(
            value["minimum_support_gain"], "minimum_support_gain",
        ),
        "maximum_support_drop": _unit(
            value["maximum_support_drop"], "maximum_support_drop",
        ),
        "minimum_conflict_margin": _unit(
            value["minimum_conflict_margin"], "minimum_conflict_margin",
        ),
        "minimum_uncontested_support": _unit(
            value["minimum_uncontested_support"], "minimum_uncontested_support",
        ),
        "minimum_consensus_raw_support": _unit(
            value["minimum_consensus_raw_support"],
            "minimum_consensus_raw_support",
        ),
    }


def _fallback(
    stage2: Mapping[str, Any], reason: str, *, phase1_rank_digest: str | None,
    policy: Mapping[str, float], branches: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "selected_output": copy.deepcopy(stage2.get("selected_output")),
        "selected_source": stage2.get("selected_source", "P0"),
        "reason": reason,
        "stage2_selected_output": copy.deepcopy(stage2.get("selected_output")),
        "stage2_selected_source": stage2.get("selected_source", "P0"),
        "phase1_rank_digest": phase1_rank_digest,
        "calibration_manifest_sha256": None,
        "policy": dict(policy),
        "branches": [] if branches is None else branches,
        "selected_branch": None,
        "used_backtrack": False,
    }
    _canonical_json(result)
    return result


def _split_audit(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    trace = row.get("method_trace")
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    if not isinstance(steps, list):
        return None
    found = [
        step.get("split_search_audit")
        for step in steps
        if isinstance(step, Mapping) and step.get("action") == "SPLIT"
        and isinstance(step.get("split_search_audit"), Mapping)
    ]
    return found[0] if len(found) == 1 else None


def _validated_branches(
    row: Mapping[str, Any], audit: Mapping[str, Any], calibration: Any,
) -> list[dict[str, Any]]:
    root = audit.get("root_ranked_siblings")
    branches = audit.get("branches")
    if not isinstance(root, list) or len(root) != 4:
        raise ValueError("split audit root ranking is invalid")
    if not isinstance(branches, list) or not 1 <= len(branches) <= 4:
        raise ValueError("split audit branch count is invalid")
    result = []
    hashes: set[str] = set()
    for index, branch in enumerate(branches):
        if not isinstance(branch, Mapping) or branch.get("visit_index") != index:
            raise ValueError("split branch visit order is invalid")
        if type(branch.get("backtracked")) is not bool:
            raise ValueError("split branch backtrack flag is invalid")
        siblings = branch.get("ranked_siblings")
        if not isinstance(siblings, list) or len(siblings) != 4:
            raise ValueError("split branch sibling ranking is invalid")
        views = []
        branch_parseable = True
        for role in ("tight", "context"):
            view = branch.get(f"{role}_view")
            if not isinstance(view, Mapping) or view.get("role") != role:
                raise ValueError("split branch view schema is invalid")
            digest = _sha256(view.get("render_sha256"), "split render hash")
            if digest in hashes:
                raise ValueError("split render hashes must be distinct")
            hashes.add(digest)
            raw_support = _unit(view.get("raw_support"), "split raw support")
            record = _answer_record(row, view.get("answer"))
            if record.aggregation_available is False or record.canonical_answer is None:
                branch_parseable = False
            views.append({
                "role": role,
                "render_sha256": digest,
                "raw_support": raw_support,
                "calibrated_support": calibration.predict(raw_support),
                "output": copy.deepcopy(record.output),
                "canonical_answer": copy.deepcopy(record.canonical_answer),
            })
        result.append({
            "visit_index": index,
            "backtracked": branch["backtracked"],
            "observed_path": copy.deepcopy(branch.get("observed_path")),
            "parseable": branch_parseable,
            "tight": views[0],
            "context": views[1],
        })
    return result


def _stage2_support(
    stage2: Mapping[str, Any], audit: Mapping[str, Any], calibration: Any,
) -> float:
    selected_source = stage2.get("selected_source")
    candidates = stage2.get("candidates")
    if isinstance(candidates, list) and candidates:
        if selected_source in {"ZOOM", "EXPAND"}:
            for candidate in candidates:
                if candidate.get("action") == selected_source:
                    return _unit(
                        candidate.get("calibrated_candidate_support"),
                        "Stage-2 selected support",
                    )
        return _unit(
            candidates[0].get("calibrated_current_support"),
            "Stage-2 current support",
        )
    p0_stability = audit.get("p0_stability")
    if not isinstance(p0_stability, Mapping):
        raise ValueError("split audit lacks P0 stability")
    return calibration.predict(_unit(
        p0_stability.get("confidence"), "P0 confidence proxy",
    ))


def _candidate_output(
    row: Mapping[str, Any], stage2_output: Any, stage2_canonical: Any,
    branch: Mapping[str, Any], confirmed_canonical: Any,
) -> Any:
    if row.get("answer_type") != "option_list":
        return copy.deepcopy(branch["tight"]["output"])
    if (
        not isinstance(stage2_output, (list, tuple))
        or stage2_canonical is None or confirmed_canonical is None
    ):
        raise ValueError("HR split projection requires an aligned semantic vote")
    tight_output = branch["tight"]["output"]
    if not isinstance(tight_output, (list, tuple)) or len(tight_output) != len(stage2_output):
        raise ValueError("HR split candidate output components differ")
    if confirmed_canonical == stage2_canonical:
        raise ValueError("HR split projection must change the semantic vote")
    return copy.deepcopy(list(tight_output))


def select_split_candidate(
    stage2_row: Mapping[str, Any], split_row: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration | None,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild frozen Stage 2, then admit only confirmed Stage 3 evidence."""
    if type(stage2_row) is not dict or type(split_row) is not dict:
        raise TypeError("split replay rows must be exact dictionaries")
    frozen_policy = _policy(policy)
    phase1_digest = _rank_digest(stage2_row)
    if phase1_digest is None or phase1_digest != _rank_digest(split_row):
        stage2 = {
            "selected_output": copy.deepcopy(stage2_row.get("output")),
            "selected_source": "P0",
        }
        return _fallback(
            stage2, "phase1_rank_drift", phase1_rank_digest=phase1_digest,
            policy=frozen_policy,
        )
    if calibration is None:
        stage2 = {
            "selected_output": copy.deepcopy(stage2_row.get("output")),
            "selected_source": "P0",
        }
        return _fallback(
            stage2, "calibration_unavailable", phase1_rank_digest=phase1_digest,
            policy=frozen_policy,
        )
    if not isinstance(calibration, (FrozenCalibration, FrozenSelectedCalibration)):
        raise TypeError("split replay calibration must be frozen")
    stage2 = replay_adaptive_search(stage2_row, split_row, calibration)
    audit = _split_audit(split_row)
    if audit is None:
        result = _fallback(
            stage2, "split_audit_unavailable", phase1_rank_digest=phase1_digest,
            policy=frozen_policy,
        )
        result["calibration_manifest_sha256"] = calibration.manifest_sha256
        return result
    try:
        if _sha256(audit.get("rank_sha256"), "split rank hash") != phase1_digest:
            raise ValueError("split rank binding differs from Phase 1")
        _sha256(audit.get("query_sha256"), "split query hash")
        branches = _validated_branches(split_row, audit, calibration)
        p0_support = _stage2_support(stage2, audit, calibration)
        stage2_projection_row = dict(split_row)
        stage2_projection_row["output"] = copy.deepcopy(stage2["selected_output"])
        stage2_canonical = _p0_canonical_answer(stage2_projection_row)
        if stage2_canonical is None:
            raise ValueError("Stage-2 output is not canonically parseable")
    except (KeyError, TypeError, ValueError):
        result = _fallback(
            stage2, "split_audit_invalid", phase1_rank_digest=phase1_digest,
            policy=frozen_policy,
        )
        result["calibration_manifest_sha256"] = calibration.manifest_sha256
        return result

    conflict_scores = [
        min(branch["tight"]["calibrated_support"],
            branch["context"]["calibrated_support"])
        for branch in branches
        if branch["parseable"]
        if branch["tight"]["canonical_answer"] == stage2_canonical
        and branch["context"]["canonical_answer"] == stage2_canonical
    ]
    branch_audits = []
    selected = None
    conflict_rejection = False
    for branch in branches:
        if not branch["parseable"]:
            branch_audits.append({
                "visit_index": branch["visit_index"],
                "observed_path": copy.deepcopy(branch["observed_path"]),
                "backtracked": branch["backtracked"],
                "raw_support": [
                    branch["tight"]["raw_support"],
                    branch["context"]["raw_support"],
                ],
                "calibrated_support": [
                    branch["tight"]["calibrated_support"],
                    branch["context"]["calibrated_support"],
                ],
                "trajectory_s": None,
                "support_gain": None,
                "selection_score": None,
                "confirmation_reason": "unparseable_view",
                "confirmed": False,
            })
            continue
        conflict_score = max(conflict_scores) if conflict_scores else None
        confirmation = confirm_split_branch(
            p0_answer=stage2_canonical,
            p0_support=p0_support,
            tight_answer=branch["tight"]["canonical_answer"],
            tight_support=branch["tight"]["calibrated_support"],
            tight_view_sha256=branch["tight"]["render_sha256"],
            context_answer=branch["context"]["canonical_answer"],
            context_support=branch["context"]["calibrated_support"],
            context_view_sha256=branch["context"]["render_sha256"],
            minimum_final_support=frozen_policy["minimum_final_support"],
            minimum_support_gain=frozen_policy["minimum_support_gain"],
            maximum_support_drop=frozen_policy["maximum_support_drop"],
            minimum_conflict_margin=frozen_policy["minimum_conflict_margin"],
            minimum_uncontested_support=frozen_policy[
                "minimum_uncontested_support"
            ],
            conflict_answer=stage2_canonical if conflict_score is not None else None,
            conflict_selection_score=conflict_score,
        )
        support_counter_rejected = (
            branch["visit_index"] >= 2
            and confirmation.confirmed
            and confirmation.reason != "confirmed_two_view_trajectory"
        )
        confirmation_reason = (
            "support_selected_counterevidence_requires_positive_gain"
            if support_counter_rejected else confirmation.reason
        )
        branch_audits.append({
            "visit_index": branch["visit_index"],
            "observed_path": copy.deepcopy(branch["observed_path"]),
            "backtracked": branch["backtracked"],
            "raw_support": [
                branch["tight"]["raw_support"],
                branch["context"]["raw_support"],
            ],
            "calibrated_support": [
                branch["tight"]["calibrated_support"],
                branch["context"]["calibrated_support"],
            ],
            "trajectory_s": confirmation.trajectory_s,
            "support_gain": confirmation.support_gain,
            "selection_score": confirmation.selection_score,
            "confirmation_reason": confirmation_reason,
            "confirmed": confirmation.confirmed and not support_counter_rejected,
        })
        conflict_rejection = conflict_rejection or confirmation.reason == (
            "equally_strong_p0_conflict"
        )
        if confirmation.confirmed and not support_counter_rejected:
            selected = (branch, confirmation)
            break

    consensus = None
    if selected is None:
        vote_groups: dict[str, dict[str, Any]] = {}
        for branch in branches:
            if not branch["parseable"]:
                continue
            for role in ("tight", "context"):
                view = branch[role]
                key = _canonical_json(view["canonical_answer"])
                group = vote_groups.setdefault(key, {
                    "canonical_answer": copy.deepcopy(view["canonical_answer"]),
                    "views": [],
                })
                group["views"].append({
                    "visit_index": branch["visit_index"],
                    "path": copy.deepcopy(branch["observed_path"]),
                    "role": role,
                    "support": view["calibrated_support"],
                    "raw_support": view["raw_support"],
                    "output": copy.deepcopy(view["output"]),
                })
        p0_key = _canonical_json(stage2_canonical)
        ordered_groups = sorted(
            vote_groups.values(),
            key=lambda group: (
                -len(group["views"]),
                _canonical_json(group["canonical_answer"]),
            ),
        )
        if ordered_groups:
            winner = ordered_groups[0]
            runner_votes = max(
                (len(group["views"]) for group in ordered_groups[1:]),
                default=0,
            )
            winner_key = _canonical_json(winner["canonical_answer"])
            distinct_paths = {
                tuple(view["path"]) for view in winner["views"]
            }
            supports = sorted(
                (view["support"] for view in winner["views"]), reverse=True,
            )
            raw_supports = sorted(
                (view["raw_support"] for view in winner["views"]), reverse=True,
            )
            support_score = supports[2] if len(supports) >= 3 else 0.0
            raw_support_score = (
                raw_supports[2] if len(raw_supports) >= 3 else 0.0
            )
            if (
                winner_key != p0_key
                and len(winner["views"]) >= 3
                and len(winner["views"]) > runner_votes
                and len(distinct_paths) >= 2
                and support_score >= frozen_policy["minimum_final_support"]
                and raw_support_score
                >= frozen_policy["minimum_consensus_raw_support"]
                and p0_support - support_score
                <= frozen_policy["maximum_support_drop"]
            ):
                best_view = max(
                    winner["views"],
                    key=lambda view: (
                        view["support"], -view["visit_index"],
                        view["role"] == "tight",
                    ),
                )
                consensus = {
                    "canonical_answer": copy.deepcopy(winner["canonical_answer"]),
                    "votes": len(winner["views"]),
                    "runner_votes": runner_votes,
                    "distinct_paths": len(distinct_paths),
                    "support_score": support_score,
                    "raw_support_score": raw_support_score,
                }
                result = {
                    "selected_output": copy.deepcopy(best_view["output"]),
                    "selected_source": "SPLIT",
                    "reason": "confirmed_cross_branch_consensus",
                    "stage2_selected_output": copy.deepcopy(stage2["selected_output"]),
                    "stage2_selected_source": stage2["selected_source"],
                    "phase1_rank_digest": phase1_digest,
                    "calibration_manifest_sha256": calibration.manifest_sha256,
                    "policy": dict(frozen_policy),
                    "branches": branch_audits,
                    "selected_branch": best_view["visit_index"],
                    "used_backtrack": best_view["visit_index"] > 0,
                    "consensus": consensus,
                }
                _canonical_json(result)
                return result

    if selected is None:
        reason = (
            "equally_strong_p0_conflict" if conflict_rejection
            else branch_audits[-1]["confirmation_reason"]
        )
        result = _fallback(
            stage2, reason, phase1_rank_digest=phase1_digest,
            policy=frozen_policy, branches=branch_audits,
        )
        result["calibration_manifest_sha256"] = calibration.manifest_sha256
        result["used_backtrack"] = len(branch_audits) > 1
        return result

    branch, confirmation = selected
    output = _candidate_output(
        split_row, stage2["selected_output"], stage2_canonical,
        branch, confirmation.canonical_answer,
    )
    result = {
        "selected_output": output,
        "selected_source": "SPLIT",
        "reason": confirmation.reason,
        "stage2_selected_output": copy.deepcopy(stage2["selected_output"]),
        "stage2_selected_source": stage2["selected_source"],
        "phase1_rank_digest": phase1_digest,
        "calibration_manifest_sha256": calibration.manifest_sha256,
        "policy": dict(frozen_policy),
        "branches": branch_audits,
        "selected_branch": branch["visit_index"],
        "used_backtrack": branch["visit_index"] > 0,
    }
    _canonical_json(result)
    return result


__all__ = ["select_split_candidate"]
