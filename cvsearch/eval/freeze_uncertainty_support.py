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

import numpy as np

from cvsearch.eval.analyze_split_search import official_correctness
from cvsearch.eval.replay_adaptive_search import (
    FrozenCalibration,
    FrozenSelectedCalibration,
)
from cvsearch.eval.replay_split_search import _split_audit

from .replay_uncertainty_support import (
    PROFILES,
    THRESHOLDS,
    AdvantageFeatures,
    RiskCalibrator,
    RiskLinearHead,
    RiskRegion,
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
    _prepare_replay,
    candidate_snapshots,
    fit_utility_isotonic,
    raw_advantage,
    replay_uncertainty_support,
    risk_basis,
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


RISK_TARGET_MODES = ("expected", "balanced", "balanced_count")
RISK_DEGREES = (1, 2)
RISK_L2 = (0.01, 0.1, 1.0, 10.0)
RISK_PENALTIES = (1.0, 2.0, 4.0)
RISK_BOUNDARIES = (-0.1, 0.0, 0.1, 0.2, 0.4, 0.6)


@dataclass(frozen=True)
class RiskModelConfiguration:
    target_mode: str
    degree: int
    l2: float
    risk_penalty: float
    decision_boundary: float


@dataclass(frozen=True)
class RiskSelection:
    configurations: tuple[tuple[str, RiskModelConfiguration], ...]
    metrics: PolicyMetrics
    folds: tuple[FoldAudit, ...]
    refit_calibrators: tuple[tuple[str, RiskCalibrator], ...]
    candidate_count: int
    source_group_count: int


@dataclass(frozen=True)
class OpenedRiskSelection:
    configurations: tuple[tuple[str, RiskModelConfiguration], ...]
    development_metrics: PolicyMetrics
    regression_metrics: PolicyMetrics
    refit_calibrators: tuple[tuple[str, RiskCalibrator], ...]
    development_candidate_count: int
    regression_candidate_count: int
    region_profiles: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _RiskExample:
    group: str
    features: AdvantageFeatures
    checkpoint: tuple[int, tuple[str, ...]]
    observations: int
    correction_units: int
    corruption_units: int
    official_units: int


@dataclass(frozen=True)
class _RiskTopic:
    record: DevelopmentRecord
    examples: tuple[_RiskExample, ...]
    stop_observations: int


LOW_SCALE_MINORITY_REGION = RiskRegion(
    lower_bounds=(0.0, 0.0, 0.005, 0.005, 0.15),
    upper_bounds=(0.0, 0.25, 0.025, 0.02, 1.0),
    expected_benefit=1.0,
    corruption_risk=0.0,
    margin=1.0,
)


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


def _risk_topics(
    records: Sequence[DevelopmentRecord],
) -> tuple[_RiskTopic, ...]:
    dummy = UnifiedPolicy(
        profile="balanced",
        threshold=0.0,
        raw_support_floor=0.0,
        utility_calibrator=UtilityIsotonicCalibrator((1.0,), (0.5,)),
    )
    result = []
    for record in records:
        stage2_row = sanitize_replay_row(record.stage2_row)
        split_row = sanitize_replay_row(record.split_row)
        audit = _split_audit(split_row)
        if (
            isinstance(audit, Mapping)
            and isinstance(audit.get("no_op_reason"), str)
            and audit["no_op_reason"]
        ):
            result.append(_RiskTopic(
                record=record, examples=(), stop_observations=0,
            ))
            continue
        prepared = _prepare_replay(stage2_row, split_row, record.calibration)
        stop_observations = sum(
            next(
                (
                    index
                    for index, view in enumerate(branch.views, start=1)
                    if view.canonical_answer is None
                ),
                len(branch.views),
            )
            for branch in prepared.branches
        )
        baseline = official_correctness(
            record.benchmark, record.stage2_row,
            prepared.stage2["selected_output"],
        )
        examples = []
        for snapshot in candidate_snapshots(
            stage2_row, split_row, record.calibration, dummy,
        ):
            candidate = official_correctness(
                record.benchmark, record.stage2_row, snapshot.output,
            )
            if len(candidate) != len(baseline):
                raise ValueError("risk candidate official units drifted")
            examples.append(_RiskExample(
                group=record.group,
                features=snapshot.features,
                checkpoint=(snapshot.branch_index, snapshot.revealed_roles),
                observations=snapshot.observations,
                correction_units=sum(
                    not old and new for old, new in zip(baseline, candidate)
                ),
                corruption_units=sum(
                    old and not new for old, new in zip(baseline, candidate)
                ),
                official_units=len(baseline),
            ))
        result.append(_RiskTopic(
            record=record,
            examples=tuple(examples),
            stop_observations=stop_observations,
        ))
    return tuple(result)


def _fit_risk_head(
    examples: Sequence[_RiskExample],
    *,
    target: str,
    target_mode: str,
    degree: int,
    l2: float,
) -> RiskLinearHead:
    if target not in {"benefit", "harm"}:
        raise ValueError("risk target must be benefit or harm")
    if target_mode not in RISK_TARGET_MODES:
        raise ValueError("unknown risk target mode")
    if degree not in RISK_DEGREES or l2 not in RISK_L2:
        raise ValueError("risk model configuration is outside the grid")
    if not examples:
        return RiskLinearHead((0.0,) * 18)
    group_counts = Counter(example.group for example in examples)
    raw_targets = []
    for example in examples:
        units = (
            example.correction_units
            if target == "benefit" else example.corruption_units
        )
        raw_targets.append(
            float(units > 0)
            if target_mode.startswith("balanced") else
            units / example.official_units
        )
    weights = np.asarray([
        1.0 / group_counts[example.group] for example in examples
    ], dtype=float)
    if target_mode.startswith("balanced"):
        class_weight = Counter()
        for weight, value in zip(weights, raw_targets):
            class_weight[value > 0.0] += (
                1.0 if target_mode == "balanced_count" else float(weight)
            )
        if class_weight[True] and class_weight[False]:
            total = class_weight[True] + class_weight[False]
            weights *= np.asarray([
                total / (2.0 * class_weight[value > 0.0])
                for value in raw_targets
            ])
    rows = [risk_basis(example.features) for example in examples]
    columns = 6 if degree == 1 else 18
    matrix = np.asarray([row[:columns] for row in rows], dtype=float)
    targets = np.asarray(raw_targets, dtype=float)
    gram = matrix.T @ (weights[:, None] * matrix)
    rhs = matrix.T @ (weights * targets)
    regularizer = np.eye(columns, dtype=float) * l2
    regularizer[0, 0] = l2 * 0.01
    try:
        coefficients = np.linalg.solve(gram + regularizer, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            gram + regularizer, rhs, rcond=None,
        )[0]
    padded = tuple(float(value) for value in coefficients) + (
        (0.0,) * (18 - columns)
    )
    return RiskLinearHead(padded)


def _fit_risk_calibrator(
    examples: Sequence[_RiskExample],
    configuration: RiskModelConfiguration,
) -> RiskCalibrator:
    if configuration.target_mode == "disabled":
        zero = RiskLinearHead((0.0,) * 18)
        return RiskCalibrator(zero, zero, 1.0, 1.0)
    return RiskCalibrator(
        benefit_head=_fit_risk_head(
            examples, target="benefit",
            target_mode=configuration.target_mode,
            degree=configuration.degree, l2=configuration.l2,
        ),
        harm_head=_fit_risk_head(
            examples, target="harm",
            target_mode=configuration.target_mode,
            degree=configuration.degree, l2=configuration.l2,
        ),
        risk_penalty=configuration.risk_penalty,
        decision_boundary=configuration.decision_boundary,
    )


def _risk_topic_decision(
    topic: _RiskTopic,
    calibrator: RiskCalibrator,
) -> tuple[int, int, int, bool, int]:
    checkpoints: list[tuple[tuple[int, tuple[str, ...]], list[_RiskExample]]] = []
    for example in topic.examples:
        if not checkpoints or checkpoints[-1][0] != example.checkpoint:
            checkpoints.append((example.checkpoint, []))
        checkpoints[-1][1].append(example)
    for _, candidates in checkpoints:
        selected = min(
            candidates,
            key=lambda example: (
                -calibrator.predict(example.features)[2],
                -example.features.agreement,
                -example.features.support,
            ),
        )
        if calibrator.predict(selected.features)[2] >= 0.0:
            return (
                selected.correction_units - selected.corruption_units,
                selected.correction_units,
                selected.corruption_units,
                True,
                selected.observations,
            )
    return 0, 0, 0, False, topic.stop_observations


def _risk_topic_outcome(
    topic: _RiskTopic,
    calibrator: RiskCalibrator,
) -> tuple[int, int, int, int]:
    delta, correction, corruption, _, observations = _risk_topic_decision(
        topic, calibrator,
    )
    return delta, correction, corruption, observations


def _risk_metrics(
    topics: Sequence[_RiskTopic],
    calibrators: Mapping[str, RiskCalibrator],
) -> PolicyMetrics:
    cell_deltas = Counter({
        f"{topic.record.backbone}/{topic.record.benchmark}": 0
        for topic in topics
    })
    dataset_deltas = Counter({topic.record.benchmark: 0 for topic in topics})
    backbone_deltas = Counter({topic.record.backbone: 0 for topic in topics})
    corrections = corruptions = observations = selections = 0
    for topic in topics:
        delta, correction, corruption, selected, observed = _risk_topic_decision(
            topic, calibrators[topic.record.group],
        )
        cell_deltas[
            f"{topic.record.backbone}/{topic.record.benchmark}"
        ] += delta
        dataset_deltas[topic.record.benchmark] += delta
        backbone_deltas[topic.record.backbone] += delta
        corrections += correction
        corruptions += corruption
        observations += observed
        selections += selected
    return PolicyMetrics(
        net_gain=corrections - corruptions,
        corrections=corrections,
        corruptions=corruptions,
        observations=observations,
        selections=selections,
        cell_deltas=tuple(sorted(cell_deltas.items())),
        dataset_deltas=tuple(sorted(dataset_deltas.items())),
        backbone_deltas=tuple(sorted(backbone_deltas.items())),
    )


def _add_metrics(
    metrics: Sequence[PolicyMetrics],
    records: Sequence[DevelopmentRecord],
) -> PolicyMetrics:
    cells = Counter({
        f"{record.backbone}/{record.benchmark}": 0 for record in records
    })
    datasets = Counter({record.benchmark: 0 for record in records})
    backbones = Counter({record.backbone: 0 for record in records})
    for item in metrics:
        cells.update(dict(item.cell_deltas))
        datasets.update(dict(item.dataset_deltas))
        backbones.update(dict(item.backbone_deltas))
    return PolicyMetrics(
        net_gain=sum(item.net_gain for item in metrics),
        corrections=sum(item.corrections for item in metrics),
        corruptions=sum(item.corruptions for item in metrics),
        observations=sum(item.observations for item in metrics),
        selections=sum(item.selections for item in metrics),
        cell_deltas=tuple(sorted(cells.items())),
        dataset_deltas=tuple(sorted(datasets.items())),
        backbone_deltas=tuple(sorted(backbones.items())),
    )


def select_risk_configuration(
    records: Sequence[DevelopmentRecord],
) -> RiskSelection:
    """Select per-stratum calibration heads by source-grouped OOF only."""
    frozen = _validate_records(records)
    topics = _risk_topics(frozen)
    strata = tuple(sorted({
        f"{topic.record.backbone}/{topic.record.stage2_row['answer_type']}"
        for topic in topics
    }))
    selected_configurations = []
    selected_metrics = []
    refit_calibrators = []
    fold_audits = []
    for stratum in strata:
        backbone, answer_type = stratum.split("/", 1)
        stratum_topics = tuple(
            topic for topic in topics
            if topic.record.backbone == backbone
            and topic.record.stage2_row["answer_type"] == answer_type
        )
        evaluation_groups = tuple(sorted({
            topic.record.group for topic in stratum_topics
        }))
        calibration_topics = stratum_topics
        if len(evaluation_groups) < 2:
            calibration_topics = topics
        calibration_groups = tuple(sorted({
            topic.record.group for topic in calibration_topics
        }))
        examples = tuple(
            example
            for topic in calibration_topics
            for example in topic.examples
        )
        if len(calibration_groups) < 2:
            raise ValueError("hierarchical risk calibration needs two groups")
        model_cache: dict[
            tuple[str, int, float, str], RiskCalibrator
        ] = {}
        for target_mode in RISK_TARGET_MODES:
            for degree in RISK_DEGREES:
                for l2 in RISK_L2:
                    for held_out in evaluation_groups:
                        training = tuple(
                            example for example in examples
                            if example.group != held_out
                        )
                        base = RiskModelConfiguration(
                            target_mode, degree, l2, 1.0, 0.0,
                        )
                        model_cache[(target_mode, degree, l2, held_out)] = (
                            _fit_risk_calibrator(training, base)
                        )
        disabled = RiskModelConfiguration(
            "disabled", 2, 10.0, 1.0, 1.0,
        )
        disabled_models = {
            group: _fit_risk_calibrator((), disabled)
            for group in evaluation_groups
        }
        zero_metrics = _risk_metrics(stratum_topics, disabled_models)
        candidates = [(
            0, 0, 0, zero_metrics.observations, 1, 0, 0, 0,
            disabled, zero_metrics,
        )]
        for mode_index, target_mode in enumerate(RISK_TARGET_MODES):
            for degree_index, degree in enumerate(RISK_DEGREES):
                for l2_index, l2 in enumerate(RISK_L2):
                    bases = {
                        group: model_cache[(target_mode, degree, l2, group)]
                        for group in evaluation_groups
                    }
                    for penalty_index, penalty in enumerate(RISK_PENALTIES):
                        for boundary_index, boundary in enumerate(RISK_BOUNDARIES):
                            configuration = RiskModelConfiguration(
                                target_mode, degree, l2, penalty, boundary,
                            )
                            calibrators = {
                                group: RiskCalibrator(
                                    base.benefit_head, base.harm_head,
                                    penalty, boundary,
                                )
                                for group, base in bases.items()
                            }
                            metrics = _risk_metrics(
                                stratum_topics, calibrators,
                            )
                            if not _feasible(metrics):
                                continue
                            candidates.append((
                                -metrics.net_gain,
                                -metrics.corrections,
                                metrics.corruptions,
                                metrics.observations,
                                0,
                                mode_index,
                                degree_index,
                                l2_index * len(RISK_PENALTIES) * len(RISK_BOUNDARIES)
                                + penalty_index * len(RISK_BOUNDARIES)
                                + boundary_index,
                                configuration,
                                metrics,
                            ))
        selected = min(candidates)
        configuration = selected[-2]
        metrics = selected[-1]
        selected_configurations.append((stratum, configuration))
        selected_metrics.append(metrics)
        refit_calibrators.append((
            stratum, _fit_risk_calibrator(examples, configuration),
        ))
        for held_out in evaluation_groups:
            fold_audits.append(FoldAudit(
                held_out_groups=(held_out,),
                train_groups=tuple(
                    group for group in calibration_groups
                    if group != held_out
                ),
                calibration_samples=sum(
                    example.group != held_out for example in examples
                ),
            ))
    zero = RiskLinearHead((0.0,) * 18)
    refit_calibrators.append((
        "*/*", RiskCalibrator(zero, zero, 1.0, 1.0),
    ))
    return RiskSelection(
        configurations=tuple(selected_configurations),
        metrics=_add_metrics(selected_metrics, frozen),
        folds=tuple(fold_audits),
        refit_calibrators=tuple(sorted(refit_calibrators)),
        candidate_count=sum(len(topic.examples) for topic in topics),
        source_group_count=len({record.group for record in frozen}),
    )


def _opened_partition_safe(metrics: PolicyMetrics) -> bool:
    return (
        metrics.corruptions == 0
        and all(delta >= 0 for _, delta in metrics.cell_deltas)
        and all(delta >= 0 for _, delta in metrics.dataset_deltas)
        and all(delta >= 0 for _, delta in metrics.backbone_deltas)
    )


def select_opened_regression_configuration(
    development_records: Sequence[DevelopmentRecord],
    regression_records: Sequence[DevelopmentRecord],
) -> OpenedRiskSelection:
    """Fit an explicitly opened-regression policy with partition safety gates."""
    development_frozen = _validate_records(development_records)
    regression_frozen = _validate_records(regression_records)
    development_topics = _risk_topics(development_frozen)
    regression_topics = _risk_topics(regression_frozen)
    all_topics = development_topics + regression_topics
    strata = tuple(sorted({
        f"{topic.record.backbone}/{topic.record.stage2_row['answer_type']}"
        for topic in all_topics
    }))
    configurations = []
    development_metrics = []
    regression_metrics = []
    calibrators = []
    region_profiles = []
    for stratum in strata:
        backbone, answer_type = stratum.split("/", 1)
        development_stratum = tuple(
            topic for topic in development_topics
            if topic.record.backbone == backbone
            and topic.record.stage2_row["answer_type"] == answer_type
        )
        regression_stratum = tuple(
            topic for topic in regression_topics
            if topic.record.backbone == backbone
            and topic.record.stage2_row["answer_type"] == answer_type
        )
        examples = tuple(
            example
            for topic in development_stratum + regression_stratum
            for example in topic.examples
        )
        disabled = RiskModelConfiguration(
            "disabled", 2, 10.0, 1.0, 1.0,
        )
        disabled_calibrator = _fit_risk_calibrator((), disabled)
        disabled_development = _risk_metrics(
            development_stratum,
            {topic.record.group: disabled_calibrator
             for topic in development_stratum},
        )
        disabled_regression = _risk_metrics(
            regression_stratum,
            {topic.record.group: disabled_calibrator
             for topic in regression_stratum},
        )
        candidates = [(
            0, 0, 0,
            disabled_development.observations
            + disabled_regression.observations,
            1, 0, 0, 0,
            disabled, disabled_calibrator,
            disabled_development, disabled_regression,
        )]
        for mode_index, target_mode in enumerate(RISK_TARGET_MODES):
            for degree_index, degree in enumerate(RISK_DEGREES):
                for l2_index, l2 in enumerate(RISK_L2):
                    base = _fit_risk_calibrator(
                        examples,
                        RiskModelConfiguration(
                            target_mode, degree, l2, 1.0, 0.0,
                        ),
                    )
                    for penalty_index, penalty in enumerate(RISK_PENALTIES):
                        for boundary_index, boundary in enumerate(RISK_BOUNDARIES):
                            configuration = RiskModelConfiguration(
                                target_mode, degree, l2, penalty, boundary,
                            )
                            calibrator = RiskCalibrator(
                                base.benefit_head, base.harm_head,
                                penalty, boundary,
                            )
                            development = _risk_metrics(
                                development_stratum,
                                {topic.record.group: calibrator
                                 for topic in development_stratum},
                            )
                            regression = _risk_metrics(
                                regression_stratum,
                                {topic.record.group: calibrator
                                 for topic in regression_stratum},
                            )
                            if not (
                                _opened_partition_safe(development)
                                and _opened_partition_safe(regression)
                            ):
                                continue
                            candidates.append((
                                -(development.net_gain + regression.net_gain),
                                -regression.net_gain,
                                -development.net_gain,
                                development.observations
                                + regression.observations,
                                0, mode_index, degree_index,
                                l2_index * len(RISK_PENALTIES) * len(RISK_BOUNDARIES)
                                + penalty_index * len(RISK_BOUNDARIES)
                                + boundary_index,
                                configuration, calibrator,
                                development, regression,
                            ))
        selected = min(candidates)
        configuration, calibrator, development, regression = selected[-4:]
        regional = RiskCalibrator(
            calibrator.benefit_head, calibrator.harm_head,
            calibrator.risk_penalty, calibrator.decision_boundary,
            regions=(LOW_SCALE_MINORITY_REGION,),
        )
        regional_development = _risk_metrics(
            development_stratum,
            {topic.record.group: regional for topic in development_stratum},
        )
        regional_regression = _risk_metrics(
            regression_stratum,
            {topic.record.group: regional for topic in regression_stratum},
        )
        base_key = (
            -(development.net_gain + regression.net_gain),
            -regression.net_gain,
            -development.net_gain,
            development.observations + regression.observations,
        )
        region_key = (
            -(regional_development.net_gain + regional_regression.net_gain),
            -regional_regression.net_gain,
            -regional_development.net_gain,
            regional_development.observations
            + regional_regression.observations,
        )
        if (
            _opened_partition_safe(regional_development)
            and _opened_partition_safe(regional_regression)
            and region_key < base_key
        ):
            calibrator = regional
            development = regional_development
            regression = regional_regression
            region_profiles.append((stratum, "low_scale_minority"))
        configurations.append((stratum, configuration))
        calibrators.append((stratum, calibrator))
        development_metrics.append(development)
        regression_metrics.append(regression)
    zero = RiskLinearHead((0.0,) * 18)
    calibrators.append((
        "*/*", RiskCalibrator(zero, zero, 1.0, 1.0),
    ))
    return OpenedRiskSelection(
        configurations=tuple(configurations),
        development_metrics=_add_metrics(
            development_metrics, development_frozen,
        ),
        regression_metrics=_add_metrics(
            regression_metrics, regression_frozen,
        ),
        refit_calibrators=tuple(sorted(calibrators)),
        development_candidate_count=sum(
            len(topic.examples) for topic in development_topics
        ),
        regression_candidate_count=sum(
            len(topic.examples) for topic in regression_topics
        ),
        region_profiles=tuple(region_profiles),
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
    """Freeze the v2 hierarchical risk policy from development only."""
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("development provenance must be nonempty")
    _reject_evaluator_payload(provenance)
    selection = select_risk_configuration(records)
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
        "schema_version": 3,
        "artifact_kind": "unified-uncertainty-support-policy",
        "data_scope": "opened_development_only",
        "selection_rule": (
            "hierarchical_source_grouped_oof_benefit_harm_risk_v2"
        ),
        "profile": "balanced",
        "weights": list(PROFILES["balanced"]),
        "threshold": 0.0,
        "raw_support_floor": 0.0,
        "utility_calibrator": UtilityIsotonicCalibrator(
            (1.0,), (0.5,),
        ).to_dict(),
        "risk_calibrators": {
            key: calibrator.to_dict()
            for key, calibrator in selection.refit_calibrators
        },
        "stratum_configurations": {
            key: {
                "target_mode": configuration.target_mode,
                "degree": configuration.degree,
                "l2": configuration.l2,
                "risk_penalty": configuration.risk_penalty,
                "decision_boundary": configuration.decision_boundary,
            }
            for key, configuration in selection.configurations
        },
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


def freeze_opened_regression_policy(
    development_records: Sequence[DevelopmentRecord],
    regression_records: Sequence[DevelopmentRecord],
    *,
    development_provenance: Mapping[str, Any],
    regression_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a post-hoc policy without representing it as unseen evidence."""
    for value in (development_provenance, regression_provenance):
        if not isinstance(value, Mapping) or not value:
            raise ValueError("opened policy provenance must be nonempty")
        _reject_evaluator_payload(value)
    selection = select_opened_regression_configuration(
        development_records, regression_records,
    )
    assignments = [
        {
            "partition": partition,
            "cell": f"{record.backbone}/{record.benchmark}",
            "ordinal": record.ordinal,
            "source_group": record.group,
        }
        for partition, records in (
            ("development", development_records),
            ("opened_regression", regression_records),
        )
        for record in sorted(
            records,
            key=lambda item: (
                item.backbone, item.benchmark, item.ordinal, item.group,
            ),
        )
    ]
    payload = {
        "schema_version": 3,
        "artifact_kind": "unified-uncertainty-support-policy",
        "data_scope": "opened_development_and_regression",
        "selection_rule": (
            "partition_safe_hierarchical_benefit_harm_regions_v2"
        ),
        "profile": "balanced",
        "weights": list(PROFILES["balanced"]),
        "threshold": 0.0,
        "raw_support_floor": 0.0,
        "utility_calibrator": UtilityIsotonicCalibrator(
            (1.0,), (0.5,),
        ).to_dict(),
        "risk_calibrators": {
            key: calibrator.to_dict()
            for key, calibrator in selection.refit_calibrators
        },
        "stratum_configurations": {
            key: {
                "target_mode": configuration.target_mode,
                "degree": configuration.degree,
                "l2": configuration.l2,
                "risk_penalty": configuration.risk_penalty,
                "decision_boundary": configuration.decision_boundary,
            }
            for key, configuration in selection.configurations
        },
        "region_profiles": dict(selection.region_profiles),
        "development_metrics": selection.development_metrics.to_dict(),
        "opened_regression_metrics": selection.regression_metrics.to_dict(),
        "development_candidate_count": (
            selection.development_candidate_count
        ),
        "opened_regression_candidate_count": (
            selection.regression_candidate_count
        ),
        "source_group_assignments_sha256": _canonical_value_hash(assignments),
        "oof_folds_sha256": _canonical_value_hash([]),
        "development_inputs": json.loads(json.dumps(
            development_provenance, sort_keys=True, allow_nan=False,
        )),
        "opened_regression_inputs": json.loads(json.dumps(
            regression_provenance, sort_keys=True, allow_nan=False,
        )),
    }
    payload["payload_sha256"] = canonical_payload_hash(payload)
    return payload
