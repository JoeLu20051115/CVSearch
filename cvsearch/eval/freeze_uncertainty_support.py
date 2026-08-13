#!/usr/bin/env python3
"""Source-grouped development freeze for the unified selector."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from cvsearch.eval.analyze_split_search import official_correctness
from cvsearch.eval.replay_adaptive_search import (
    FrozenCalibration,
    FrozenSelectedCalibration,
)

from .replay_uncertainty_support import (
    PROFILES,
    THRESHOLDS,
    AdvantageFeatures,
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
    _prepare_replay,
    candidate_snapshots,
    fit_utility_isotonic,
    raw_advantage,
    replay_uncertainty_support,
    sanitize_replay_row,
)


BENCHMARKS = frozenset({"hr_bench_4k", "hr_bench_8k", "treebench", "vstar"})


class NoFeasibleConfiguration(ValueError):
    """Raised when development OOF cannot satisfy the declared no-harm gate."""


def source_group(benchmark: str, ordinal: int, input_image: str) -> str:
    """Return one source identity shared across backbones and HR resolutions."""
    if benchmark not in BENCHMARKS:
        raise ValueError("unsupported unified-selector benchmark")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("source ordinal must be a nonnegative exact integer")
    if not isinstance(input_image, str) or not input_image.strip():
        raise ValueError("source image identity must be nonempty")
    if benchmark.startswith("hr_bench_"):
        return f"hr_bench:{ordinal}"
    normalized = str(PurePosixPath(input_image.strip()))
    if normalized in {"", "."}:
        raise ValueError("source image identity must be nonempty")
    return f"{benchmark}:{normalized}"


def utility_target(delta: int, official_units: int) -> float:
    """Map candidate-vs-P0 official-unit delta around neutral utility 0.5."""
    if type(delta) is not int or type(official_units) is not int:
        raise TypeError("utility target inputs must be exact integers")
    if official_units <= 0 or abs(delta) > official_units:
        raise ValueError("official utility target is invalid")
    return 0.5 + delta / (2 * official_units)


@dataclass(frozen=True)
class DevelopmentRecord:
    """One labeled development topic kept outside the frozen policy."""

    group: str
    backbone: str
    benchmark: str
    ordinal: int
    stage2_row: Mapping[str, Any]
    split_row: Mapping[str, Any]
    calibration: FrozenCalibration | FrozenSelectedCalibration

    def __post_init__(self) -> None:
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("development source group must be nonempty")
        if not isinstance(self.backbone, str) or not self.backbone:
            raise ValueError("development backbone must be nonempty")
        if self.benchmark not in BENCHMARKS:
            raise ValueError("development benchmark is unsupported")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("development ordinal must be nonnegative")
        if type(self.stage2_row) is not dict or type(self.split_row) is not dict:
            raise TypeError("development replay rows must be exact dictionaries")
        if self.stage2_row.get("_eg_ordinal") != self.ordinal:
            raise ValueError("Stage-2 development ordinal is misbound")
        if self.split_row.get("_eg_ordinal") != self.ordinal:
            raise ValueError("SPLIT development ordinal is misbound")
        if not isinstance(
            self.calibration, (FrozenCalibration, FrozenSelectedCalibration),
        ):
            raise TypeError("development support calibration must be frozen")


@dataclass(frozen=True)
class FoldAudit:
    held_out_groups: tuple[str, ...]
    train_groups: tuple[str, ...]
    calibration_samples: int


@dataclass(frozen=True)
class PolicyMetrics:
    net_gain: int
    corrections: int
    corruptions: int
    observations: int
    selections: int
    cell_deltas: tuple[tuple[str, int], ...]
    dataset_deltas: tuple[tuple[str, int], ...]
    backbone_deltas: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "net_gain": self.net_gain,
            "corrections": self.corrections,
            "corruptions": self.corruptions,
            "observations": self.observations,
            "selections": self.selections,
            "cell_deltas": dict(self.cell_deltas),
            "dataset_deltas": dict(self.dataset_deltas),
            "backbone_deltas": dict(self.backbone_deltas),
        }


@dataclass(frozen=True)
class Selection:
    profile: str
    threshold: float
    metrics: PolicyMetrics
    folds: tuple[FoldAudit, ...]
    refit_calibrator: UtilityIsotonicCalibrator
    candidate_count: int
    source_group_count: int


def _validate_records(
    records: Sequence[DevelopmentRecord],
) -> tuple[DevelopmentRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("development records must be a sequence")
    result = tuple(records)
    if len({record.group for record in result}) < 2:
        raise ValueError("grouped OOF requires at least two source groups")
    seen = set()
    for record in result:
        if not isinstance(record, DevelopmentRecord):
            raise TypeError("development records must be frozen records")
        key = (record.backbone, record.benchmark, record.ordinal)
        if key in seen:
            raise ValueError("development cells must have unique ordinals")
        seen.add(key)
    return result


def _candidate_examples(
    record: DevelopmentRecord,
    *,
    raw_support_floor: float,
) -> tuple[tuple[AdvantageFeatures, float], ...]:
    dummy = UnifiedPolicy(
        profile="balanced",
        threshold=0.0,
        raw_support_floor=raw_support_floor,
        utility_calibrator=UtilityIsotonicCalibrator((1.0,), (0.5,)),
    )
    stage2_policy_row = sanitize_replay_row(record.stage2_row)
    split_policy_row = sanitize_replay_row(record.split_row)
    prepared = _prepare_replay(
        stage2_policy_row, split_policy_row, record.calibration,
    )
    baseline = official_correctness(
        record.benchmark, record.stage2_row,
        prepared.stage2["selected_output"],
    )
    result = []
    for snapshot in candidate_snapshots(
        stage2_policy_row, split_policy_row, record.calibration, dummy,
    ):
        candidate = official_correctness(
            record.benchmark, record.stage2_row, snapshot.output,
        )
        if len(candidate) != len(baseline):
            raise ValueError("development candidate official units drifted")
        delta = sum(candidate) - sum(baseline)
        result.append((
            snapshot.features,
            utility_target(delta, len(baseline)),
        ))
    return tuple(result)


def _fit_fold(
    records: Sequence[DevelopmentRecord],
    examples: Mapping[int, tuple[tuple[AdvantageFeatures, float], ...]],
    profile: str,
    held_out: str,
) -> tuple[UtilityIsotonicCalibrator, int]:
    samples = [
        (raw_advantage(features, PROFILES[profile]), target)
        for index, record in enumerate(records)
        if record.group != held_out
        for features, target in examples[index]
    ]
    if not samples:
        raise ValueError("grouped OOF training fold has no candidate snapshots")
    return fit_utility_isotonic(tuple(samples)), len(samples)


def _metrics(
    records: Sequence[DevelopmentRecord],
    calibrators: Mapping[tuple[str, str], UtilityIsotonicCalibrator],
    *,
    profile: str,
    threshold: float,
    raw_support_floor: float,
) -> PolicyMetrics:
    cell_deltas: Counter[str] = Counter()
    dataset_deltas: Counter[str] = Counter()
    backbone_deltas: Counter[str] = Counter()
    corrections = corruptions = observations = selections = 0
    for record in records:
        policy = UnifiedPolicy(
            profile=profile,
            threshold=threshold,
            raw_support_floor=raw_support_floor,
            utility_calibrator=calibrators[(profile, record.group)],
        )
        decision = replay_uncertainty_support(
            sanitize_replay_row(record.stage2_row),
            sanitize_replay_row(record.split_row),
            record.calibration, policy,
        )
        before = official_correctness(
            record.benchmark, record.stage2_row,
            decision["stage2_selected_output"],
        )
        after = official_correctness(
            record.benchmark, record.stage2_row,
            decision["selected_output"],
        )
        if len(before) != len(after):
            raise ValueError("development replay official units drifted")
        delta = sum(after) - sum(before)
        cell_deltas[f"{record.backbone}/{record.benchmark}"] += delta
        dataset_deltas[record.benchmark] += delta
        backbone_deltas[record.backbone] += delta
        corrections += sum(not old and new for old, new in zip(before, after))
        corruptions += sum(old and not new for old, new in zip(before, after))
        observations += decision["observations"]
        selections += decision["selected_source"] == "SPLIT"
    return PolicyMetrics(
        net_gain=sum(cell_deltas.values()),
        corrections=corrections,
        corruptions=corruptions,
        observations=observations,
        selections=selections,
        cell_deltas=tuple(sorted(cell_deltas.items())),
        dataset_deltas=tuple(sorted(dataset_deltas.items())),
        backbone_deltas=tuple(sorted(backbone_deltas.items())),
    )


def _feasible(metrics: PolicyMetrics) -> bool:
    return (
        all(delta >= 0 for _, delta in metrics.cell_deltas)
        and all(delta >= 0 for _, delta in metrics.dataset_deltas)
        and all(delta >= 0 for _, delta in metrics.backbone_deltas)
        and metrics.corrections > metrics.corruptions
    )


def select_configuration(
    records: Sequence[DevelopmentRecord],
    *,
    raw_support_floor: float = 0.2,
) -> Selection:
    """Choose one global profile/threshold by leave-one-source-group-out replay."""
    frozen = _validate_records(records)
    raw_support_floor = float(raw_support_floor)
    if not 0.0 <= raw_support_floor <= 1.0:
        raise ValueError("raw support floor must be in [0, 1]")
    examples = {
        index: _candidate_examples(
            record, raw_support_floor=raw_support_floor,
        )
        for index, record in enumerate(frozen)
    }
    groups = tuple(sorted({record.group for record in frozen}))
    fold_calibrators = {}
    fold_counts = {}
    for profile in PROFILES:
        for held_out in groups:
            calibrator, count = _fit_fold(
                frozen, examples, profile, held_out,
            )
            fold_calibrators[(profile, held_out)] = calibrator
            fold_counts[(profile, held_out)] = count

    candidates = []
    for profile_index, profile in enumerate(PROFILES):
        for threshold_index, threshold in enumerate(THRESHOLDS):
            metrics = _metrics(
                frozen, fold_calibrators, profile=profile,
                threshold=threshold, raw_support_floor=raw_support_floor,
            )
            if _feasible(metrics):
                candidates.append((
                    -metrics.net_gain,
                    metrics.corruptions,
                    metrics.observations,
                    profile_index,
                    threshold_index,
                    profile,
                    threshold,
                    metrics,
                ))
    if not candidates:
        raise NoFeasibleConfiguration(
            "no source-grouped configuration satisfies every no-harm gate",
        )
    selected = min(candidates)
    profile, threshold, metrics = selected[5:]
    all_samples = [
        (raw_advantage(features, PROFILES[profile]), target)
        for record_examples in examples.values()
        for features, target in record_examples
    ]
    refit = fit_utility_isotonic(tuple(all_samples))
    folds = tuple(FoldAudit(
        held_out_groups=(held_out,),
        train_groups=tuple(group for group in groups if group != held_out),
        calibration_samples=fold_counts[(profile, held_out)],
    ) for held_out in groups)
    return Selection(
        profile=profile,
        threshold=threshold,
        metrics=metrics,
        folds=folds,
        refit_calibrator=refit,
        candidate_count=len(all_samples),
        source_group_count=len(groups),
    )


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash a payload after excluding its self-authentication field."""
    if not isinstance(payload, Mapping):
        raise TypeError("frozen policy payload must be a mapping")
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_value_hash(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reject_evaluator_payload(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("answer", "correct", "target", "label")):
                raise ValueError("policy provenance must not contain evaluator labels")
            _reject_evaluator_payload(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_evaluator_payload(child)


def freeze_policy(
    records: Sequence[DevelopmentRecord],
    *,
    provenance: Mapping[str, Any],
    raw_support_floor: float = 0.2,
) -> dict[str, Any]:
    """Freeze a deterministic policy containing aggregates and hashes only."""
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("development provenance must be nonempty")
    _reject_evaluator_payload(provenance)
    selection = select_configuration(
        records, raw_support_floor=raw_support_floor,
    )
    assignments = [
        {
            "cell": f"{record.backbone}/{record.benchmark}",
            "ordinal": record.ordinal,
            "source_group": record.group,
        }
        for record in sorted(
            records,
            key=lambda item: (
                item.backbone, item.benchmark, item.ordinal, item.group,
            ),
        )
    ]
    folds = [
        {
            "held_out_groups": list(fold.held_out_groups),
            "train_groups": list(fold.train_groups),
            "calibration_samples": fold.calibration_samples,
        }
        for fold in selection.folds
    ]
    payload = {
        "schema_version": 2,
        "artifact_kind": "unified-uncertainty-support-policy",
        "data_scope": "opened_development_only",
        "selection_rule": "leave_one_source_group_out_no_harm_utility_v1",
        "profile": selection.profile,
        "weights": list(PROFILES[selection.profile]),
        "threshold": selection.threshold,
        "raw_support_floor": float(raw_support_floor),
        "utility_calibrator": selection.refit_calibrator.to_dict(),
        "topic_count": len(records),
        "source_group_count": selection.source_group_count,
        "source_group_assignments_sha256": _canonical_value_hash(assignments),
        "oof_folds_sha256": _canonical_value_hash(folds),
        "candidate_count": selection.candidate_count,
        "oof_metrics": selection.metrics.to_dict(),
        "development_inputs": json.loads(json.dumps(
            provenance, sort_keys=True, allow_nan=False,
        )),
    }
    payload["payload_sha256"] = canonical_payload_hash(payload)
    return payload
