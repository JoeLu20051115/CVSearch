"""Evaluator-only Stage 3b support-calibration extraction and freezing."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .replay_adaptive_search import freeze_selected_calibration
from .treebench_support_proxy import treebench_geometry_support_label
from .vstar_support_proxy import vstar_geometry_support_label


_POLICIES = {
    "native_2x2_overlap_support_screen_two_scale_depth2_v2",
    "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3",
}


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _split_audit(row: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = row.get("method_trace")
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    matches = [
        step.get("split_search_audit")
        for step in steps if isinstance(step, Mapping)
        and step.get("action") == "SPLIT"
        and isinstance(step.get("split_search_audit"), Mapping)
    ] if isinstance(steps, list) else []
    if len(matches) != 1:
        raise ValueError("calibration row requires exactly one SPLIT audit")
    return matches[0]


def extract_split_support_rows(
    benchmark: str, rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract support-only grouped rows from opened geometric annotations."""
    if benchmark not in {"vstar", "treebench"}:
        raise ValueError("SPLIT geometry calibration supports only V* and TreeBench")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("SPLIT calibration observations must be a sequence")
    labeler = (
        vstar_geometry_support_label
        if benchmark == "vstar" else treebench_geometry_support_label
    )
    extracted = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("SPLIT calibration rows must be mappings")
        if row.get("split") not in {None, "dev", "development"}:
            raise ValueError("SPLIT calibration may use only development rows")
        ordinal = row.get("_eg_ordinal")
        source = row.get("input_image")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("SPLIT calibration ordinal is invalid")
        if type(source) is not str or not source:
            raise ValueError("SPLIT calibration source group is invalid")
        audit = _split_audit(row)
        if audit.get("render_policy") not in _POLICIES:
            raise ValueError("SPLIT calibration render policy is not frozen")
        branches = audit.get("branches")
        if not isinstance(branches, list) or not branches:
            raise ValueError("SPLIT calibration audit has no branches")
        for expected_index, branch in enumerate(branches):
            if (
                not isinstance(branch, Mapping)
                or branch.get("visit_index") != expected_index
            ):
                raise ValueError("SPLIT calibration branch order is invalid")
            roles = ["tight", "context"]
            if "medium_view" in branch:
                roles.insert(1, "medium")
            for role in roles:
                view = branch.get(f"{role}_view")
                if not isinstance(view, Mapping) or view.get("role") != role:
                    raise ValueError("SPLIT calibration view role is invalid")
                crop = view.get("crop_xyxy")
                row_id = f"{benchmark}:{ordinal}:{expected_index}:{role}"
                if row_id in seen:
                    raise ValueError("SPLIT calibration row identity is duplicated")
                seen.add(row_id)
                extracted.append({
                    "row_id": row_id,
                    "source_group": f"{benchmark}:{source}",
                    "raw_support": _unit(
                        view.get("raw_support"), "SPLIT raw support",
                    ),
                    "support_sufficient": labeler(row, [crop]),
                })
    return extracted


def split_geometry_ordering_metrics(
    benchmark: str, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure answer-free GT evidence ranks on opened development rows."""
    if benchmark not in {"vstar", "treebench"}:
        raise ValueError("SPLIT ordering metrics support only V* and TreeBench")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("SPLIT ordering metrics require nonempty observations")
    labeler = (
        vstar_geometry_support_label
        if benchmark == "vstar" else treebench_geometry_support_label
    )
    first_roots: list[int | None] = []
    first_branches: list[int | None] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("SPLIT ordering rows must be mappings")
        if row.get("split") not in {None, "dev", "development"}:
            raise ValueError("SPLIT ordering metrics may use only development rows")
        audit = _split_audit(row)
        if audit.get("render_policy") != (
            "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
        ):
            raise ValueError("SPLIT ordering metrics require the all-root policy")
        roots = audit.get("root_ranked_siblings")
        branches = audit.get("branches")
        if not isinstance(roots, list) or len(roots) != 4:
            raise ValueError("SPLIT ordering root ranking must contain four roots")
        if not isinstance(branches, list) or len(branches) != 6:
            raise ValueError("SPLIT ordering branch ranking must contain six branches")
        root_labels = []
        for root in roots:
            if not isinstance(root, Mapping):
                raise ValueError("SPLIT ordering root is invalid")
            root_labels.append(labeler(row, [root.get("box")]))
        branch_labels = []
        for index, branch in enumerate(branches):
            if not isinstance(branch, Mapping) or branch.get("visit_index") != index:
                raise ValueError("SPLIT ordering branch visit order is invalid")
            tight = branch.get("tight_view")
            if not isinstance(tight, Mapping):
                raise ValueError("SPLIT ordering tight view is invalid")
            branch_labels.append(labeler(row, [tight.get("crop_xyxy")]))
        first_roots.append(next(
            (index + 1 for index, value in enumerate(root_labels) if value), None,
        ))
        first_branches.append(next(
            (index + 1 for index, value in enumerate(branch_labels) if value), None,
        ))

    def recall(values: Sequence[int | None], limit: int) -> int:
        return sum(value is not None and value <= limit for value in values)

    return {
        "topics": len(rows),
        "root_recall_at_1": recall(first_roots, 1),
        "root_recall_at_2": recall(first_roots, 2),
        "root_recall_at_4": recall(first_roots, 4),
        "branch_recall_at_1": recall(first_branches, 1),
        "branch_recall_at_4": recall(first_branches, 4),
        "branch_recall_at_6": recall(first_branches, 6),
        "first_root_evidence_rank": dict(sorted(Counter(
            "missing" if value is None else str(value) for value in first_roots
        ).items())),
        "first_branch_evidence_rank": dict(sorted(Counter(
            "missing" if value is None else str(value) for value in first_branches
        ).items())),
        "rescue_only_recovered": sum(
            value is not None and 4 < value <= 6 for value in first_branches
        ),
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def freeze_split_calibration_suite(
    observations: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    """Freeze independent backbone mappings without serializing evaluator truth."""
    if not isinstance(observations, Mapping) or not observations:
        raise ValueError("SPLIT calibration suite requires backbone observations")
    backbones = {}
    for backbone in sorted(observations):
        datasets = observations[backbone]
        if type(backbone) is not str or not backbone or not isinstance(datasets, Mapping):
            raise ValueError("SPLIT calibration backbone record is invalid")
        support_rows = []
        for benchmark in sorted(datasets):
            support_rows.extend(extract_split_support_rows(
                benchmark, datasets[benchmark],
            ))
        calibration = freeze_selected_calibration(support_rows)
        backbones[backbone] = calibration.to_dict()
    payload = {
        "schema_version": 1,
        "artifact_kind": "stage3b-split-support-calibration",
        "data_scope": "opened_development_only",
        "label_contract": (
            "evaluator-only V*/TreeBench geometry; labels absent from inference"
        ),
        "selection_rule": (
            "leave-one-source-group-out Brier/ECE over frozen shrinkage grid"
        ),
        "backbones": backbones,
    }
    return dict(payload, suite_sha256=_canonical_sha256(payload))


__all__ = [
    "extract_split_support_rows", "freeze_split_calibration_suite",
    "split_geometry_ordering_metrics",
]
