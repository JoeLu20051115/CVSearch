"""Evaluator-only fixed-pool SPLIT ranking ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from PIL import Image

from cvsearch.evidence_gap.method import _split_visual_features
from cvsearch.evidence_gap.split_search import (
    SPLIT_RELEVANCE_WEIGHT,
    SplitPatch,
    rank_split_children,
)

from .treebench_support_proxy import treebench_geometry_support_label
from .vstar_support_proxy import vstar_geometry_support_label
from .analyze_split_search import candidate_outputs, official_correctness


_ALL_ROOT_POLICY = (
    "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3"
)
_POLICIES = ("grid", "visual_only", "clip_only", "combined")
_LABELERS = {
    "vstar": vstar_geometry_support_label,
    "treebench": treebench_geometry_support_label,
}


def _split_audit(row: Mapping[str, Any]) -> Mapping[str, Any]:
    trace = row.get("method_trace")
    steps = trace.get("steps") if isinstance(trace, Mapping) else None
    matches = [
        step.get("split_search_audit")
        for step in steps
        if isinstance(step, Mapping)
        and step.get("action") == "SPLIT"
        and isinstance(step.get("split_search_audit"), Mapping)
    ] if isinstance(steps, list) else []
    if len(matches) != 1:
        raise ValueError("ranking row requires exactly one SPLIT audit")
    if matches[0].get("render_policy") != _ALL_ROOT_POLICY:
        raise ValueError("ranking row must use the frozen all-root policy")
    return matches[0]


def _finite_unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _path(value: Any, depth: int, name: str) -> tuple[int, ...]:
    if (
        type(value) is not list
        or len(value) != depth
        or any(type(item) is not int or not 0 <= item < 4 for item in value)
    ):
        raise ValueError(f"{name} must be an exact depth-{depth} split path")
    return tuple(value)


def _box(value: Any, image: Image.Image, name: str) -> tuple[int, int, int, int]:
    if (
        type(value) is not list
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"{name} must be an exact integer XYXY box")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError(f"{name} is outside the source image")
    return tuple(value)


def _components(
    image: Image.Image,
    records: Sequence[Mapping[str, Any]],
    *,
    depth: int,
    box_key: str,
    score_key: str,
) -> dict[tuple[int, ...], dict[str, float]]:
    if len(records) != 4:
        raise ValueError("each split sibling set must contain four candidates")
    patches = []
    combined = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("split candidate records must be mappings")
        path = _path(record.get("path") if depth == 1 else record.get(
            "patch_path"
        ), depth, f"split candidate {index} path")
        box = _box(record.get(box_key), image, f"split candidate {index} box")
        patches.append(SplitPatch(path, box))
        combined[path] = _finite_unit(
            record.get(score_key), f"split candidate {index} combined score",
        )
    if len({patch.path for patch in patches}) != 4 or len({patch.box for patch in patches}) != 4:
        raise ValueError("split sibling candidates must have unique identities")
    edges, deviations = _split_visual_features(image, patches)
    visual = {
        item.patch.path: item.visual_information
        for item in rank_split_children(patches, [0.0] * 4, edges, deviations)
    }
    result = {}
    for path in combined:
        clip = (
            combined[path] - (1 - SPLIT_RELEVANCE_WEIGHT) * visual[path]
        ) / SPLIT_RELEVANCE_WEIGHT
        if not -1e-9 <= clip <= 1 + 1e-9:
            raise ValueError("frozen combined score cannot recover a CLIP percentile")
        result[path] = {
            "visual_only": visual[path],
            "clip_only": min(1.0, max(0.0, clip)),
            "combined": combined[path],
        }
    return result


def fixed_pool_scores(
    row: Mapping[str, Any], image: Image.Image,
) -> dict[tuple[int, int], dict[str, float]]:
    """Recover three answer-free ranking scores for the exact sixteen leaves."""
    if not isinstance(row, Mapping) or not isinstance(image, Image.Image):
        raise TypeError("fixed-pool reconstruction requires a row and PIL image")
    audit = _split_audit(row)
    roots = audit.get("root_ranked_siblings")
    probes = audit.get("screening_probes")
    if not isinstance(roots, list) or len(roots) != 4:
        raise ValueError("fixed-pool ranking requires four roots")
    if not isinstance(probes, list) or len(probes) != 16:
        raise ValueError("fixed-pool ranking requires exactly sixteen probes")
    root_components = _components(
        image, roots, depth=1, box_key="box", score_key="score",
    )
    probe_paths = [
        _path(probe.get("patch_path"), 2, f"split probe {index} path")
        if isinstance(probe, Mapping) else ()
        for index, probe in enumerate(probes)
    ]
    if len(set(probe_paths)) != 16:
        raise ValueError("fixed-pool probes must have unique candidate identities")
    if {path[:1] for path in probe_paths} != set(root_components):
        raise ValueError("fixed-pool probes do not cover all roots")
    leaf_components = {}
    for root_path in root_components:
        siblings = [
            probe for probe, path in zip(probes, probe_paths)
            if path[:1] == root_path
        ]
        children = _components(
            image, siblings, depth=2, box_key="target_box_xyxy",
            score_key="ranking_score",
        )
        for path, child in children.items():
            leaf_components[path] = {
                name: (root_components[root_path][name] + child[name]) / 2
                for name in ("visual_only", "clip_only", "combined")
            }
    if len(leaf_components) != 16:
        raise ValueError("fixed-pool reconstruction did not produce sixteen leaves")
    return leaf_components


def _ks(ks: Sequence[int], count: int) -> tuple[int, ...]:
    if (
        isinstance(ks, (str, bytes))
        or not isinstance(ks, Sequence)
        or not ks
        or any(type(value) is not int or not 1 <= value <= count for value in ks)
        or tuple(sorted(set(ks))) != tuple(ks)
    ):
        raise ValueError("K values must be unique ascending integers within the pool")
    return tuple(ks)


def exact_random_metrics(
    labels: Sequence[bool], ks: Sequence[int],
) -> dict[str, Any]:
    """Return exact expectations under a uniformly random leaf permutation."""
    if (
        isinstance(labels, (str, bytes))
        or not isinstance(labels, Sequence)
        or not labels
        or any(type(value) is not bool for value in labels)
    ):
        raise ValueError("random ranking labels must be a nonempty boolean sequence")
    count = len(labels)
    limits = _ks(ks, count)
    positives = sum(labels)
    recall = {}
    for limit in limits:
        misses = (
            math.comb(count - positives, limit) / math.comb(count, limit)
            if count - positives >= limit else 0.0
        )
        recall[str(limit)] = 1.0 - misses
    distribution = {}
    mrr = 0.0
    if positives:
        for rank in range(1, count - positives + 2):
            probability = 1.0
            for prior in range(rank - 1):
                probability *= (count - positives - prior) / (count - prior)
            probability *= positives / (count - rank + 1)
            distribution[str(rank)] = probability
            mrr += probability / rank
    else:
        distribution["missing"] = 1.0
    return {
        "recall_at": recall,
        "pool_recall": float(positives > 0),
        "mrr": mrr,
        "mean_first_evidence_rank": (
            (count + 1) / (positives + 1) if positives else None
        ),
        "first_evidence_rank": distribution,
    }


def _ordered_metrics(
    labels: Mapping[tuple[int, int], bool],
    order: Sequence[tuple[int, int]],
    ks: Sequence[int],
) -> dict[str, Any]:
    first = next((index + 1 for index, path in enumerate(order) if labels[path]), None)
    return {
        "recall_at": {
            str(limit): float(first is not None and first <= limit) for limit in ks
        },
        "pool_recall": float(first is not None),
        "mrr": 0.0 if first is None else 1 / first,
        "mean_first_evidence_rank": first,
        "first_evidence_rank": {"missing" if first is None else str(first): 1.0},
    }


def _aggregate(topic_metrics: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    count = len(topic_metrics)
    distributions: Counter[str] = Counter()
    for metrics in topic_metrics:
        distributions.update(metrics["first_evidence_rank"])
    observed_ranks = [
        metrics["mean_first_evidence_rank"]
        for metrics in topic_metrics
        if metrics["mean_first_evidence_rank"] is not None
    ]
    return {
        "recall_at": {
            str(limit): sum(metrics["recall_at"][str(limit)] for metrics in topic_metrics) / count
            for limit in ks
        },
        "pool_recall": sum(metrics["pool_recall"] for metrics in topic_metrics) / count,
        "mrr": sum(metrics["mrr"] for metrics in topic_metrics) / count,
        "mean_first_evidence_rank": (
            sum(observed_ranks) / len(observed_ranks) if observed_ranks else None
        ),
        "first_evidence_rank": dict(sorted(distributions.items())),
    }


def _source_identity(row: Mapping[str, Any]) -> str:
    return json.dumps({
        "ordinal": row.get("_eg_ordinal"),
        "input_image": row.get("input_image"),
        "question": row.get("question"),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _candidate_geometry(row: Mapping[str, Any]) -> tuple[Any, ...]:
    audit = _split_audit(row)
    roots = audit.get("root_ranked_siblings")
    probes = audit.get("screening_probes")
    if not isinstance(roots, list) or not isinstance(probes, list):
        raise ValueError("fixed-pool candidate geometry is missing")
    return (
        tuple(sorted(
            (_path(item.get("path"), 1, "root path"), tuple(item.get("box", ())))
            for item in roots if isinstance(item, Mapping)
        )),
        tuple(sorted(
            (
                _path(item.get("patch_path"), 2, "probe path"),
                tuple(item.get("target_box_xyxy", ())),
            )
            for item in probes if isinstance(item, Mapping)
        )),
    )


def evaluate_fixed_pool(
    observations_by_backbone: Mapping[
        str, Mapping[str, Sequence[Mapping[str, Any]]]
    ],
    image_roots: Mapping[str, Path],
    ks: Sequence[int] = (1, 3, 6),
) -> dict[str, Any]:
    """Evaluate fixed-pool rankings once after verifying backbone identity."""
    if not isinstance(observations_by_backbone, Mapping) or len(
        observations_by_backbone
    ) < 2:
        raise ValueError("fixed-pool evaluation requires at least two backbones")
    backbones = sorted(observations_by_backbone)
    datasets = sorted(observations_by_backbone[backbones[0]])
    if not datasets or set(datasets) - set(_LABELERS):
        raise ValueError("fixed-pool ranking supports only V* and TreeBench")
    if any(sorted(observations_by_backbone[name]) != datasets for name in backbones):
        raise ValueError("backbone ranking datasets do not align")
    limits = _ks(ks, 16)
    by_dataset_topics: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        dataset: {name: [] for name in ("random_expected", *_POLICIES)}
        for dataset in datasets
    }
    by_backbone_topics = {
        backbone: {name: [] for name in ("random_expected", *_POLICIES)}
        for backbone in backbones
    }
    source_topic_count = 0
    evaluation_count = 0
    for dataset in datasets:
        indexed = {}
        for backbone in backbones:
            rows = observations_by_backbone[backbone][dataset]
            if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
                raise TypeError("backbone observations must be sequences")
            index = {row.get("_eg_ordinal"): row for row in rows}
            if len(index) != len(rows) or any(type(value) is not int for value in index):
                raise ValueError("backbone ordinals must be unique exact integers")
            indexed[backbone] = index
        ordinals = sorted(indexed[backbones[0]])
        if not ordinals or any(sorted(indexed[name]) != ordinals for name in backbones):
            raise ValueError("backbone ranking ordinals do not align")
        root = Path(image_roots[dataset])
        for ordinal in ordinals:
            rows = [indexed[name][ordinal] for name in backbones]
            identity = _source_identity(rows[0])
            if any(_source_identity(row) != identity for row in rows[1:]):
                raise ValueError("cross-backbone source identity drifted")
            geometry = _candidate_geometry(rows[0])
            if any(_candidate_geometry(row) != geometry for row in rows[1:]):
                raise ValueError("cross-backbone candidate geometry drifted")
            source = rows[0].get("input_image")
            if not isinstance(source, str) or not source:
                raise ValueError("ranking input image must be nonempty")
            with Image.open(root / source) as opened:
                image = opened.convert("RGB")
            audit = _split_audit(rows[0])
            boxes = {
                _path(probe.get("patch_path"), 2, "split probe path"):
                probe.get("target_box_xyxy")
                for probe in audit["screening_probes"]
            }
            labeler = _LABELERS[dataset]
            labels = {
                path: bool(labeler(rows[0], [box])) for path, box in boxes.items()
            }
            for backbone, row in zip(backbones, rows):
                scores = fixed_pool_scores(row, image)
                random_metrics = exact_random_metrics(list(labels.values()), limits)
                by_dataset_topics[dataset]["random_expected"].append(random_metrics)
                by_backbone_topics[backbone]["random_expected"].append(random_metrics)
                orders = {"grid": sorted(scores)}
                for policy in ("visual_only", "clip_only", "combined"):
                    orders[policy] = sorted(
                        scores, key=lambda path: (-scores[path][policy], path),
                    )
                for policy, order in orders.items():
                    metrics = _ordered_metrics(labels, order, limits)
                    by_dataset_topics[dataset][policy].append(metrics)
                    by_backbone_topics[backbone][policy].append(metrics)
                evaluation_count += 1
            source_topic_count += 1
    by_dataset = {
        dataset: {
            policy: _aggregate(metrics, limits)
            for policy, metrics in policies.items()
        }
        for dataset, policies in by_dataset_topics.items()
    }
    policies = {
        policy: _aggregate([
            metrics
            for dataset in datasets
            for metrics in by_dataset_topics[dataset][policy]
        ], limits)
        for policy in ("random_expected", *_POLICIES)
    }
    by_backbone = {
        backbone: {
            policy: _aggregate(metrics, limits)
            for policy, metrics in policies_by_backbone.items()
        }
        for backbone, policies_by_backbone in by_backbone_topics.items()
    }
    combined = policies["combined"]
    gates = {
        "combined_recall_at_3_above_random": combined["recall_at"]["3"]
        > policies["random_expected"]["recall_at"]["3"],
        "combined_recall_at_3_above_grid": combined["recall_at"]["3"]
        > policies["grid"]["recall_at"]["3"],
        "combined_recall_at_3_at_least_single_signals": all(
            combined["recall_at"]["3"] >= policies[name]["recall_at"]["3"]
            for name in ("visual_only", "clip_only")
        ),
        "combined_recall_at_6_at_least_single_signals": all(
            combined["recall_at"]["6"] >= policies[name]["recall_at"]["6"]
            for name in ("visual_only", "clip_only")
        ),
        "combined_mrr_at_least_single_signals": all(
            combined["mrr"] >= policies[name]["mrr"]
            for name in ("visual_only", "clip_only")
        ),
        "combined_recall_at_3_no_dataset_grid_regression": all(
            by_dataset[dataset]["combined"]["recall_at"]["3"]
            >= by_dataset[dataset]["grid"]["recall_at"]["3"]
            for dataset in datasets
        ),
    }
    return {
        "source_topics": source_topic_count,
        "topic_backbone_evaluations": evaluation_count,
        "candidate_pool_size": 16,
        "backbone_copies_verified": len(backbones),
        "topic_copy_pairs_verified": source_topic_count,
        "ks": list(limits),
        "policies": policies,
        "by_backbone": by_backbone,
        "by_dataset": by_dataset,
        "gates": gates,
        "success": all(gates.values()),
    }


_FAILURE_COUNTS = (
    "official_units", "stage2_correct", "stage2_errors", "stage3b_correct",
    "converted", "selector_abstained", "selector_wrong_choice",
    "no_correct_observed_answer", "geometry_pool_miss",
    "six_branch_budget_miss", "vlm_answer_miss", "corruption",
)


def _failure_counter() -> Counter[str]:
    return Counter({name: 0 for name in _FAILURE_COUNTS})


def _failure_geometry(
    benchmark: str, row: Mapping[str, Any],
) -> str | None:
    if benchmark not in _LABELERS:
        return None
    audit = _split_audit(row)
    probes = audit.get("screening_probes")
    branches = audit.get("branches")
    if not isinstance(probes, list) or len(probes) != 16:
        raise ValueError("failure geometry requires sixteen fixed probes")
    if not isinstance(branches, list) or len(branches) != 6:
        raise ValueError("failure geometry requires six observed branches")
    labeler = _LABELERS[benchmark]
    pool_crops = [probe.get("target_box_xyxy") for probe in probes]
    branch_crops = []
    for index, branch in enumerate(branches):
        if not isinstance(branch, Mapping) or branch.get("visit_index") != index:
            raise ValueError("failure geometry branch order is invalid")
        tight = branch.get("tight_view")
        if not isinstance(tight, Mapping):
            raise ValueError("failure geometry tight view is missing")
        branch_crops.append(tight.get("crop_xyxy"))
    if not any(labeler(row, [crop]) for crop in pool_crops):
        return "geometry_pool_miss"
    if not any(labeler(row, [crop]) for crop in branch_crops):
        return "six_branch_budget_miss"
    return "vlm_answer_miss"


def _count_dict(counter: Counter[str]) -> dict[str, int]:
    return {name: counter[name] for name in _FAILURE_COUNTS}


def decompose_frozen_failures(
    score_report: Mapping[str, Any],
    observations_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Partition frozen validation failures without replaying or tuning policy."""
    if not isinstance(score_report, Mapping) or not isinstance(
        observations_by_cell, Mapping
    ):
        raise TypeError("failure decomposition requires report and cell observations")
    cells = score_report.get("cells")
    aggregate = score_report.get("aggregate")
    if not isinstance(cells, Mapping) or not cells or not isinstance(aggregate, Mapping):
        raise ValueError("frozen score report is incomplete")
    if set(cells) != set(observations_by_cell):
        raise ValueError("frozen score and observation cells do not align")
    total = _failure_counter()
    cell_counts: dict[str, Counter[str]] = {}
    backbone_counts: dict[str, Counter[str]] = {}
    dataset_counts: dict[str, Counter[str]] = {}
    row_count = 0
    recomputed_corrections = 0
    recomputed_corruptions = 0
    geometry_refined = 0
    for cell in sorted(cells):
        try:
            backbone, benchmark = cell.split("/", 1)
        except ValueError as error:
            raise ValueError("frozen cell identity must be backbone/dataset") from error
        report_cell = cells[cell]
        decisions = report_cell.get("rows") if isinstance(report_cell, Mapping) else None
        raw_rows = observations_by_cell[cell]
        if not isinstance(decisions, list) or isinstance(raw_rows, (str, bytes)) or not isinstance(
            raw_rows, Sequence
        ):
            raise ValueError("frozen cell rows are invalid")
        indexed = {row.get("_eg_ordinal"): row for row in raw_rows}
        if len(indexed) != len(raw_rows) or any(type(value) is not int for value in indexed):
            raise ValueError("frozen observation ordinals must be unique integers")
        decision_index = {row.get("ordinal"): row for row in decisions}
        if len(decision_index) != len(decisions) or set(indexed) != set(decision_index):
            raise ValueError("frozen decision and observation ordinals do not align")
        counter = _failure_counter()
        backbone_counter = backbone_counts.setdefault(backbone, _failure_counter())
        dataset_counter = dataset_counts.setdefault(benchmark, _failure_counter())
        for ordinal in sorted(indexed):
            raw = indexed[ordinal]
            decision = decision_index[ordinal]
            before = tuple(decision.get("stage2_correct", ()))
            after = tuple(decision.get("stage3b_correct", ()))
            if (
                not before or len(before) != len(after)
                or any(type(value) is not bool for value in (*before, *after))
            ):
                raise ValueError("frozen correctness flags are invalid")
            recomputed_before = official_correctness(
                benchmark, raw, decision.get("stage2_output"),
            )
            recomputed_after = official_correctness(
                benchmark, raw, decision.get("stage3b_output"),
            )
            if before != recomputed_before or after != recomputed_after:
                raise ValueError("frozen correctness differs from raw benchmark labels")
            candidates = tuple(
                official_correctness(benchmark, raw, output)
                for output in candidate_outputs(raw)
            )
            if any(len(candidate) != len(before) for candidate in candidates):
                raise ValueError("candidate official-unit count drifted")
            selected_source = decision.get("selected_source")
            if selected_source not in {"P0", "ZOOM", "EXPAND", "SPLIT"}:
                raise ValueError("frozen selected source is invalid")
            if (
                selected_source != "SPLIT"
                and decision.get("stage2_output") != decision.get("stage3b_output")
            ):
                raise ValueError("non-SPLIT fallback changed the frozen Stage-2 output")
            for index, (old, new) in enumerate(zip(before, after)):
                counter["official_units"] += 1
                counter["stage2_correct"] += old
                counter["stage2_errors"] += not old
                counter["stage3b_correct"] += new
                if old:
                    if not new:
                        counter["corruption"] += 1
                        recomputed_corruptions += 1
                    continue
                correct_candidate = any(candidate[index] for candidate in candidates)
                if new:
                    counter["converted"] += 1
                    recomputed_corrections += 1
                elif correct_candidate and selected_source != "SPLIT":
                    counter["selector_abstained"] += 1
                elif correct_candidate:
                    counter["selector_wrong_choice"] += 1
                else:
                    counter["no_correct_observed_answer"] += 1
                    geometry = _failure_geometry(benchmark, raw)
                    if geometry is not None:
                        counter[geometry] += 1
                        geometry_refined += 1
            row_count += 1
        cell_counts[cell] = counter
        total.update(counter)
        backbone_counter.update(counter)
        dataset_counter.update(counter)
    expected = {
        "topics": row_count,
        "official_units": total["official_units"],
        "stage2_correct": total["stage2_correct"],
        "stage3b_correct": total["stage3b_correct"],
        "corrections": recomputed_corrections,
        "corruptions": recomputed_corruptions,
    }
    for name, value in expected.items():
        if aggregate.get(name) != value:
            raise ValueError(f"frozen aggregate {name} does not reconcile")
    gates = {
        "stage2_errors_partition_exactly": (
            total["converted"] + total["selector_abstained"]
            + total["selector_wrong_choice"]
            + total["no_correct_observed_answer"]
            == total["stage2_errors"]
        ),
        "geometry_refinements_partition_geometry_failures": (
            total["geometry_pool_miss"] + total["six_branch_budget_miss"]
            + total["vlm_answer_miss"] == geometry_refined
        ),
        "corrections_match_frozen_report": recomputed_corrections
        == aggregate.get("corrections"),
        "corruptions_match_frozen_report": recomputed_corruptions
        == aggregate.get("corruptions"),
        "official_units_match_frozen_report": total["official_units"]
        == aggregate.get("official_units"),
    }
    if not all(gates.values()):
        raise ValueError("frozen failure decomposition did not reconcile")
    return {
        "aggregate": _count_dict(total),
        "cells": {cell: _count_dict(value) for cell, value in cell_counts.items()},
        "by_backbone": {
            name: _count_dict(value) for name, value in backbone_counts.items()
        },
        "by_dataset": {
            name: _count_dict(value) for name, value in dataset_counts.items()
        },
        "gates": gates,
        "success": all(gates.values()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if type(row) is not dict:
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-pool SPLIT rankings and frozen failures.",
    )
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--vstar-image-root", type=Path, required=True)
    parser.add_argument("--treebench-image-root", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--validation-split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict CPU-only evaluator and atomically write canonical JSON."""
    args = _parser().parse_args(argv)
    image_roots = {
        "vstar": args.vstar_image_root,
        "treebench": args.treebench_image_root,
    }
    backbones = sorted(
        path.name for path in args.development_root.iterdir()
        if path.is_dir()
        and all((path / f"{dataset}.jsonl").is_file() for dataset in _LABELERS)
    )
    if len(backbones) < 2:
        raise ValueError("development root must contain at least two backbones")
    observations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    development_hashes = {}
    image_hashes = {}
    for backbone in backbones:
        observations[backbone] = {}
        for dataset in sorted(_LABELERS):
            path = args.development_root / backbone / f"{dataset}.jsonl"
            rows = _jsonl(path)
            observations[backbone][dataset] = rows
            development_hashes[f"{backbone}/{dataset}"] = _sha256(path)
            for row in rows:
                source = row.get("input_image")
                if not isinstance(source, str) or not source:
                    raise ValueError("development input image identity is invalid")
                key = f"{dataset}/{source}"
                image_hashes.setdefault(key, _sha256(image_roots[dataset] / source))
    ranking = evaluate_fixed_pool(observations, image_roots)
    try:
        validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("validation report is not valid JSON") from error
    if type(validation_report) is not dict or not isinstance(
        validation_report.get("cells"), Mapping
    ):
        raise ValueError("validation report has no frozen cells")
    validation_observations = {}
    validation_hashes = {}
    for cell in sorted(validation_report["cells"]):
        try:
            backbone, dataset = cell.split("/", 1)
        except ValueError as error:
            raise ValueError("validation cell must be backbone/dataset") from error
        path = args.validation_split_root / backbone / f"{dataset}.jsonl"
        validation_observations[cell] = _jsonl(path)
        validation_hashes[cell] = _sha256(path)
    failures = decompose_frozen_failures(
        validation_report, validation_observations,
    )
    gates = {
        "ranking_success": ranking["success"],
        "failure_decomposition_success": failures["success"],
        "fixed_candidate_pool_is_sixteen": ranking["candidate_pool_size"] == 16,
        "validation_is_posthoc_only": True,
        "all_source_hashes_bound": all((
            development_hashes, image_hashes, validation_hashes,
        )),
    }
    payload = {
        "schema_version": 1,
        "artifact_kind": "fixed-pool-split-ranking-ablation",
        "data_scope": {
            "ranking": "opened_development_vstar_treebench_only",
            "failure_decomposition": "frozen_validation_v3_posthoc_only",
            "validation_used_for_ranking_or_tuning": False,
        },
        "bindings": {
            "development_jsonl_sha256": development_hashes,
            "development_image_sha256": image_hashes,
            "validation_report_sha256": _sha256(args.validation_report),
            "validation_split_jsonl_sha256": validation_hashes,
        },
        "ranking": ranking,
        "failure_decomposition": failures,
        "gates": gates,
        "success": all(gates.values()),
    }
    _write_canonical_json(args.output, payload)
    return 0


__all__ = [
    "decompose_frozen_failures", "evaluate_fixed_pool",
    "exact_random_metrics", "fixed_pool_scores", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
