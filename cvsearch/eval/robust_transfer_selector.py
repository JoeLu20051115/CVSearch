#!/usr/bin/env python3
"""Auditable acceptance gates for robust Stage-3 policy transfer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .freeze_uncertainty_support import (
    BENCHMARKS,
    RISK_BOUNDARIES,
    RISK_DEGREES,
    RISK_L2,
    RISK_PENALTIES,
    RISK_TARGET_MODES,
    DevelopmentRecord,
    FoldAudit,
    PolicyMetrics,
    RiskModelConfiguration,
    _RiskTopic,
    _fit_risk_calibrator,
    _risk_topic_decision,
    _risk_topics,
)
from .replay_uncertainty_support import (
    RiskCalibrator,
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
)


BACKBONES = ("internvl", "qwen")
REQUIRED_CELLS = frozenset(
    f"{backbone}/{benchmark}"
    for backbone in BACKBONES
    for benchmark in BENCHMARKS
)


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Frozen development gates; zero corruption remains a preference."""

    max_mean_observations: float = 12.8
    preferred_mean_observations: float = 11.52
    minimum_net_gain: int = 10

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_mean_observations, (int, float))
            or isinstance(self.max_mean_observations, bool)
            or self.max_mean_observations <= 0.0
        ):
            raise ValueError("maximum mean observations must be positive")
        if (
            not isinstance(self.preferred_mean_observations, (int, float))
            or isinstance(self.preferred_mean_observations, bool)
            or not 0.0 < self.preferred_mean_observations
            <= self.max_mean_observations
        ):
            raise ValueError("preferred observation cost must be within the cap")
        if (
            type(self.minimum_net_gain) is not int
            or self.minimum_net_gain < 0
        ):
            raise ValueError("minimum net gain must be a nonnegative integer")


def _validated(
    metrics: PolicyMetrics, topics: int,
) -> tuple[dict[str, int], dict[str, int], float]:
    if not isinstance(metrics, PolicyMetrics):
        raise TypeError("robust transfer metrics must be PolicyMetrics")
    if type(topics) is not int or topics <= 0:
        raise ValueError("topic count must be a positive exact integer")
    return (
        dict(metrics.backbone_deltas),
        dict(metrics.cell_deltas),
        metrics.observations / topics,
    )


def evaluate_acceptance(
    metrics: PolicyMetrics,
    topics: int,
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
) -> tuple[str, ...]:
    """Return deterministic reasons that a development report cannot promote."""
    if not isinstance(criteria, AcceptanceCriteria):
        raise TypeError("acceptance criteria must be frozen")
    backbones, cells, mean_observations = _validated(metrics, topics)
    failures = []
    if (
        set(backbones) != set(BACKBONES)
        or len(metrics.backbone_deltas) != len(BACKBONES)
    ):
        failures.append("required backbones are incomplete")
    if (
        set(cells) != REQUIRED_CELLS
        or len(metrics.cell_deltas) != len(REQUIRED_CELLS)
    ):
        failures.append("required cells are incomplete")
    failures.extend(
        f"{cell} cell is negative"
        for cell, delta in sorted(cells.items())
        if delta < 0
    )
    failures.extend(
        f"{backbone} backbone is not strictly positive"
        for backbone in BACKBONES
        if backbones.get(backbone, 0) <= 0
    )
    if metrics.net_gain < criteria.minimum_net_gain:
        failures.append(
            f"net gain is below {criteria.minimum_net_gain}"
        )
    if metrics.corrections <= metrics.corruptions:
        failures.append("corrections do not exceed corruptions")
    if mean_observations > criteria.max_mean_observations:
        failures.append(
            "mean observations exceed "
            f"{criteria.max_mean_observations:g}"
        )
    return tuple(failures)


def robust_rank(
    metrics: PolicyMetrics,
    topics: int,
    grid_index: int,
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
) -> tuple[Any, ...]:
    """Return a stable ascending key led by feasibility and worst-cell safety."""
    if type(grid_index) is not int or grid_index < 0:
        raise ValueError("grid index must be a nonnegative exact integer")
    backbones, cells, mean_observations = _validated(metrics, topics)
    failures = evaluate_acceptance(metrics, topics, criteria)
    complete_backbones = set(backbones) == set(BACKBONES)
    complete_cells = set(cells) == REQUIRED_CELLS
    worst_backbone = min(backbones.values(), default=-10**9)
    worst_cell = min(cells.values(), default=-10**9)
    return (
        bool(failures),
        metrics.corrections <= metrics.corruptions,
        not complete_cells or worst_cell < 0,
        not complete_backbones or worst_backbone <= 0,
        metrics.net_gain < criteria.minimum_net_gain,
        mean_observations > criteria.max_mean_observations,
        metrics.corruptions != 0,
        -worst_backbone,
        -worst_cell,
        -metrics.net_gain,
        mean_observations > criteria.preferred_mean_observations,
        metrics.corruptions,
        metrics.observations,
        -metrics.corrections,
        grid_index,
    )


@dataclass(frozen=True)
class SharedRiskSelection:
    """One globally configured action rule selected by source-group OOF."""

    configuration: RiskModelConfiguration
    metrics: PolicyMetrics
    failures: tuple[str, ...]
    folds: tuple[FoldAudit, ...]
    refit_calibrators: tuple[tuple[str, RiskCalibrator], ...]
    candidate_count: int
    source_group_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": _configuration_dict(self.configuration),
            "metrics": self.metrics.to_dict(),
            "failures": list(self.failures),
            "folds": [
                {
                    "held_out_groups": list(fold.held_out_groups),
                    "train_groups": list(fold.train_groups),
                    "calibration_samples": fold.calibration_samples,
                }
                for fold in self.folds
            ],
            "refit_calibrators": {
                key: calibrator.to_dict()
                for key, calibrator in self.refit_calibrators
            },
            "candidate_count": self.candidate_count,
            "source_group_count": self.source_group_count,
        }


@dataclass(frozen=True)
class OuterPartitionFold:
    held_out_partition: str
    train_partitions: tuple[str, ...]
    held_out_groups: tuple[str, ...]
    train_groups: tuple[str, ...]
    configuration: RiskModelConfiguration
    inner_metrics: PolicyMetrics
    inner_failures: tuple[str, ...]
    metrics: PolicyMetrics
    policy: UnifiedPolicy = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_partition": self.held_out_partition,
            "train_partitions": list(self.train_partitions),
            "held_out_groups": list(self.held_out_groups),
            "train_groups": list(self.train_groups),
            "configuration": _configuration_dict(self.configuration),
            "inner_metrics": self.inner_metrics.to_dict(),
            "inner_failures": list(self.inner_failures),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class NestedTransferSelection:
    outer_folds: tuple[OuterPartitionFold, ...]
    combined_oof_metrics: PolicyMetrics
    failures: tuple[str, ...]
    refit_selection: SharedRiskSelection | None
    refit_policy: UnifiedPolicy | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_folds": [fold.to_dict() for fold in self.outer_folds],
            "combined_oof_metrics": self.combined_oof_metrics.to_dict(),
            "failures": list(self.failures),
            "refit_selection": (
                None
                if self.refit_selection is None
                else self.refit_selection.to_dict()
            ),
            "refit_policy": (
                None
                if self.refit_policy is None
                else {
                    "profile": self.refit_policy.profile,
                    "threshold": self.refit_policy.threshold,
                    "raw_support_floor": self.refit_policy.raw_support_floor,
                    "risk_calibrators": {
                        key: calibrator.to_dict()
                        for key, calibrator
                        in self.refit_policy.risk_calibrators
                    },
                }
            ),
        }


def _configuration_dict(
    configuration: RiskModelConfiguration,
) -> dict[str, Any]:
    return {
        "target_mode": configuration.target_mode,
        "degree": configuration.degree,
        "l2": configuration.l2,
        "risk_penalty": configuration.risk_penalty,
        "decision_boundary": configuration.decision_boundary,
    }


def _validate_shared_records(
    records: Sequence[DevelopmentRecord],
) -> tuple[DevelopmentRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("shared-selector records must be a sequence")
    frozen = tuple(records)
    if len({record.group for record in frozen}) < 2:
        raise ValueError("grouped OOF requires at least two source groups")
    seen = set()
    for record in frozen:
        if not isinstance(record, DevelopmentRecord):
            raise TypeError("shared-selector records must be frozen records")
        identity = (
            record.group, record.backbone, record.benchmark, record.ordinal,
        )
        if identity in seen:
            raise ValueError("shared-selector records must be unique")
        seen.add(identity)
    return frozen


def _stratum(topic: _RiskTopic) -> str:
    return (
        f"{topic.record.backbone}/"
        f"{topic.record.stage2_row['answer_type']}"
    )


def _bind_action(
    calibrator: RiskCalibrator, penalty: float, boundary: float,
) -> RiskCalibrator:
    return RiskCalibrator(
        benefit_head=calibrator.benefit_head,
        harm_head=calibrator.harm_head,
        risk_penalty=penalty,
        decision_boundary=boundary,
    )


def _fit_base_heads(
    topics: Sequence[_RiskTopic],
    base: RiskModelConfiguration,
    *,
    excluded_group: str | None = None,
) -> dict[str, RiskCalibrator]:
    strata = tuple(sorted({_stratum(topic) for topic in topics}))
    result = {}
    for key in (*strata, "*/*"):
        examples = tuple(
            example
            for topic in topics
            if key == "*/*" or _stratum(topic) == key
            for example in topic.examples
            if example.group != excluded_group
        )
        result[key] = _fit_risk_calibrator(examples, base)
    return result


def _with_action(
    heads: Mapping[str, RiskCalibrator],
    penalty: float,
    boundary: float,
) -> dict[str, RiskCalibrator]:
    return {
        key: _bind_action(calibrator, penalty, boundary)
        for key, calibrator in heads.items()
    }


def _metrics_for_topics(
    topics: Sequence[_RiskTopic],
    calibrator_for: Callable[[_RiskTopic], RiskCalibrator],
) -> PolicyMetrics:
    cells = Counter({
        f"{topic.record.backbone}/{topic.record.benchmark}": 0
        for topic in topics
    })
    datasets = Counter({topic.record.benchmark: 0 for topic in topics})
    backbones = Counter({topic.record.backbone: 0 for topic in topics})
    corrections = corruptions = observations = selections = 0
    for topic in topics:
        delta, correction, corruption, selected, observed = (
            _risk_topic_decision(
                topic, calibrator_for(topic),
            )
        )
        cells[f"{topic.record.backbone}/{topic.record.benchmark}"] += delta
        datasets[topic.record.benchmark] += delta
        backbones[topic.record.backbone] += delta
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
        cell_deltas=tuple(sorted(cells.items())),
        dataset_deltas=tuple(sorted(datasets.items())),
        backbone_deltas=tuple(sorted(backbones.items())),
    )


def _calibrator_for(
    calibrators: Mapping[str, RiskCalibrator], topic: _RiskTopic,
) -> RiskCalibrator:
    return calibrators.get(_stratum(topic), calibrators["*/*"])


def _sum_metrics(
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


def select_shared_configuration(
    records: Sequence[DevelopmentRecord],
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
) -> SharedRiskSelection:
    """Select one configuration with leave-one-source-group-out replay."""
    frozen = _validate_shared_records(records)
    topics = _risk_topics(frozen)
    groups = tuple(sorted({record.group for record in frozen}))
    bases = tuple(
        RiskModelConfiguration(mode, degree, l2, 1.0, 0.0)
        for mode in RISK_TARGET_MODES
        for degree in RISK_DEGREES
        for l2 in RISK_L2
    )
    cached = {
        (base_index, held_out): _fit_base_heads(
            topics, base, excluded_group=held_out,
        )
        for base_index, base in enumerate(bases)
        for held_out in groups
    }
    candidates = []
    grid_index = 0
    for base_index, base in enumerate(bases):
        for penalty in RISK_PENALTIES:
            for boundary in RISK_BOUNDARIES:
                fold_calibrators = {
                    held_out: _with_action(
                        cached[(base_index, held_out)], penalty, boundary,
                    )
                    for held_out in groups
                }
                oof_metrics = _metrics_for_topics(
                    topics,
                    lambda topic, lookup=fold_calibrators: _calibrator_for(
                        lookup[topic.record.group], topic,
                    ),
                )
                configuration = RiskModelConfiguration(
                    base.target_mode, base.degree, base.l2,
                    penalty, boundary,
                )
                candidates.append((
                    robust_rank(
                        oof_metrics, len(topics), grid_index, criteria,
                    ),
                    configuration,
                    oof_metrics,
                ))
                grid_index += 1
    _, selected, selected_metrics = min(candidates, key=lambda item: item[0])
    refit = _with_action(
        _fit_base_heads(topics, selected),
        selected.risk_penalty,
        selected.decision_boundary,
    )
    folds = tuple(
        FoldAudit(
            held_out_groups=(held_out,),
            train_groups=tuple(group for group in groups if group != held_out),
            calibration_samples=sum(
                example.group != held_out
                for topic in topics
                for example in topic.examples
            ),
        )
        for held_out in groups
    )
    return SharedRiskSelection(
        configuration=selected,
        metrics=selected_metrics,
        failures=evaluate_acceptance(
            selected_metrics, len(topics), criteria,
        ),
        folds=folds,
        refit_calibrators=tuple(sorted(refit.items())),
        candidate_count=sum(len(topic.examples) for topic in topics),
        source_group_count=len(groups),
    )


def _policy(
    calibrators: Sequence[tuple[str, RiskCalibrator]],
) -> UnifiedPolicy:
    return UnifiedPolicy(
        profile="balanced",
        threshold=0.0,
        raw_support_floor=0.0,
        utility_calibrator=UtilityIsotonicCalibrator((1.0,), (0.5,)),
        risk_calibrators=tuple(calibrators),
    )


def nested_partition_validation(
    partitions: Mapping[str, Sequence[DevelopmentRecord]],
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
) -> NestedTransferSelection:
    """Estimate shared-config selection with outer partition isolation."""
    if not isinstance(partitions, Mapping) or len(partitions) < 2:
        raise ValueError("nested validation requires at least two partitions")
    validated = {}
    for name, records in sorted(partitions.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("partition names must be nonempty strings")
        validated[name] = _validate_shared_records(records)
    folds = []
    outer_metrics = []
    all_records = tuple(
        record
        for name in sorted(validated)
        for record in validated[name]
    )
    for held_out_name in sorted(validated):
        held_out = validated[held_out_name]
        held_out_groups = frozenset(record.group for record in held_out)
        train_names = tuple(
            name for name in sorted(validated) if name != held_out_name
        )
        train = tuple(
            record
            for name in train_names
            for record in validated[name]
            if record.group not in held_out_groups
        )
        selected = select_shared_configuration(train, criteria)
        train_groups = tuple(sorted({record.group for record in train}))
        held_out_topics = _risk_topics(held_out)
        refit = dict(selected.refit_calibrators)
        metrics = _metrics_for_topics(
            held_out_topics,
            lambda topic, lookup=refit: _calibrator_for(lookup, topic),
        )
        outer_metrics.append(metrics)
        folds.append(OuterPartitionFold(
            held_out_partition=held_out_name,
            train_partitions=train_names,
            held_out_groups=tuple(sorted(held_out_groups)),
            train_groups=train_groups,
            configuration=selected.configuration,
            inner_metrics=selected.metrics,
            inner_failures=selected.failures,
            metrics=metrics,
            policy=_policy(selected.refit_calibrators),
        ))
    combined = _sum_metrics(outer_metrics, all_records)
    failures = evaluate_acceptance(combined, len(all_records), criteria)
    refit_selection = None
    refit_policy = None
    if not failures:
        refit_selection = select_shared_configuration(all_records, criteria)
        if refit_selection.failures:
            failures = tuple(
                f"opened refit: {failure}"
                for failure in refit_selection.failures
            )
        else:
            refit_policy = _policy(refit_selection.refit_calibrators)
    return NestedTransferSelection(
        outer_folds=tuple(folds),
        combined_oof_metrics=combined,
        failures=failures,
        refit_selection=refit_selection,
        refit_policy=refit_policy,
    )
