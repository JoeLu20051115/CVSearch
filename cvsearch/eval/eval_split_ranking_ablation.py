"""Evaluator-only fixed-pool SPLIT ranking ablations."""

from __future__ import annotations

import json
import math
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


__all__ = ["evaluate_fixed_pool", "exact_random_metrics", "fixed_pool_scores"]
