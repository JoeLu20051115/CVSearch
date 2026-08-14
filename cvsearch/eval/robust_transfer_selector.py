#!/usr/bin/env python3
"""Auditable acceptance gates for robust Stage-3 policy transfer."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .analyze_split_search import official_correctness
from .candidate_free_verifier_selector import (
    CandidateFreeVerifierEvidence,
    candidate_free_cascade_outcome,
    candidate_free_outcome_for_selected,
)

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
    AggregateEvidenceFeatures,
    RiskCalibrator,
    RiskLinearHead,
    RiskLogisticHead,
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
    _field_values,
)


BACKBONES = ("internvl", "qwen")
REQUIRED_CELLS = frozenset(
    f"{backbone}/{benchmark}"
    for backbone in BACKBONES
    for benchmark in BENCHMARKS
)
EVIDENCE_FEATURE_MODES = {
    "base": tuple(range(5)),
    "mean": tuple(range(7)),
    "topology": tuple(range(14)),
}
EVIDENCE_BALANCING = (False, True)
MINIMUM_OBSERVATIONS = (1, 2, 4, 6, 8)
MAXIMUM_OBSERVATIONS = (8, 10, 11, 12, 14)
MINIMUM_AGREEING_VIEWS = 2
REFERENCE_OFFICIAL_UNITS = 256
PROPOSAL_VERIFIER_CONFIDENCES = (0.6, 0.7, 0.8, 0.9)
VERIFIER_PROPOSAL_AGREEMENTS = (0.4, 0.5, 0.6)


@dataclass(frozen=True)
class AggregateRiskConfiguration:
    """One global aggregate-risk model and shared runtime action rule."""

    feature_mode: str
    regularization: float
    balanced: bool
    risk_penalty: float
    decision_boundary: float
    minimum_observations: int
    maximum_observations: int
    minimum_agreeing_views: int = MINIMUM_AGREEING_VIEWS
    proposal_verifier_confidence: float | None = None
    verifier_proposal_agreement: float | None = None

    def __post_init__(self) -> None:
        if self.feature_mode not in EVIDENCE_FEATURE_MODES:
            raise ValueError("unknown aggregate evidence feature mode")
        if self.regularization not in RISK_L2:
            raise ValueError("aggregate regularization is outside the grid")
        if type(self.balanced) is not bool:
            raise TypeError("aggregate class balancing must be an exact boolean")
        if self.risk_penalty not in RISK_PENALTIES:
            raise ValueError("aggregate risk penalty is outside the grid")
        if self.decision_boundary not in RISK_BOUNDARIES:
            raise ValueError("aggregate decision boundary is outside the grid")
        if self.minimum_observations not in MINIMUM_OBSERVATIONS:
            raise ValueError("minimum observations are outside the grid")
        if self.maximum_observations not in MAXIMUM_OBSERVATIONS:
            raise ValueError("maximum observations are outside the grid")
        if self.minimum_observations > self.maximum_observations:
            raise ValueError("minimum observations exceed maximum observations")
        if self.minimum_agreeing_views != MINIMUM_AGREEING_VIEWS:
            raise ValueError("aggregate selector requires two-view confirmation")
        verifier_values = (
            self.proposal_verifier_confidence,
            self.verifier_proposal_agreement,
        )
        if (verifier_values[0] is None) != (verifier_values[1] is None):
            raise ValueError("verifier action thresholds must be paired")
        if verifier_values[0] is not None and (
            verifier_values[0] not in PROPOSAL_VERIFIER_CONFIDENCES
            or verifier_values[1] not in VERIFIER_PROPOSAL_AGREEMENTS
        ):
            raise ValueError("verifier action threshold is outside the grid")


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


def _scaled_criteria(
    criteria: AcceptanceCriteria,
    records: Sequence[DevelopmentRecord],
) -> AcceptanceCriteria:
    """Scale the +10/256 recall gate to the exact official-unit pool."""
    units = sum(
        len(official_correctness(
            record.benchmark, record.stage2_row,
            record.stage2_row.get("output"),
        ))
        for record in records
    )
    return AcceptanceCriteria(
        max_mean_observations=criteria.max_mean_observations,
        preferred_mean_observations=criteria.preferred_mean_observations,
        minimum_net_gain=math.ceil(
            criteria.minimum_net_gain * units / REFERENCE_OFFICIAL_UNITS
        ),
    )


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
    backbone_gap = (
        max(backbones.values()) - worst_backbone
        if backbones else 10**9
    )
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
        backbone_gap,
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

    configuration: RiskModelConfiguration | AggregateRiskConfiguration
    metrics: PolicyMetrics
    failures: tuple[str, ...]
    folds: tuple[FoldAudit, ...]
    refit_calibrators: tuple[tuple[str, RiskCalibrator], ...]
    candidate_count: int
    source_group_count: int
    verifier_cascade: bool

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
            "verifier_cascade": self.verifier_cascade,
        }


@dataclass(frozen=True)
class OuterPartitionFold:
    held_out_partition: str
    train_partitions: tuple[str, ...]
    held_out_groups: tuple[str, ...]
    train_groups: tuple[str, ...]
    configuration: RiskModelConfiguration | AggregateRiskConfiguration
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
    configuration: RiskModelConfiguration | AggregateRiskConfiguration,
) -> dict[str, Any]:
    if isinstance(configuration, AggregateRiskConfiguration):
        result = {
            "feature_mode": configuration.feature_mode,
            "regularization": configuration.regularization,
            "balanced": configuration.balanced,
            "risk_penalty": configuration.risk_penalty,
            "decision_boundary": configuration.decision_boundary,
            "minimum_observations": configuration.minimum_observations,
            "maximum_observations": configuration.maximum_observations,
            "minimum_agreeing_views": configuration.minimum_agreeing_views,
        }
        if configuration.proposal_verifier_confidence is not None:
            result.update({
                "proposal_verifier_confidence": (
                    configuration.proposal_verifier_confidence
                ),
                "verifier_proposal_agreement": (
                    configuration.verifier_proposal_agreement
                ),
            })
        return result
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


def verifier_evidence_key(
    record: DevelopmentRecord,
) -> tuple[str, str, str, int]:
    """Return the exact identity joining one record to verifier evidence."""
    if not isinstance(record, DevelopmentRecord):
        raise TypeError("verifier evidence key requires a development record")
    return (
        record.group, record.backbone, record.benchmark, record.ordinal,
    )


def _verifier_lookup(
    topics: Sequence[_RiskTopic],
    evidence: Mapping[
        tuple[str, str, str, int], CandidateFreeVerifierEvidence
    ] | None,
) -> dict[tuple[str, str, str, int], CandidateFreeVerifierEvidence] | None:
    if evidence is None:
        return None
    if not isinstance(evidence, Mapping):
        raise TypeError("verifier evidence must be an identity mapping")
    result = {}
    for topic in topics:
        key = verifier_evidence_key(topic.record)
        value = evidence.get(key)
        if not isinstance(value, CandidateFreeVerifierEvidence):
            raise ValueError("verifier evidence is missing a bound topic")
        result[key] = value
    return result


def _stratum(topic: _RiskTopic) -> str:
    return (
        f"{topic.record.backbone}/"
        f"{topic.record.stage2_row['answer_type']}"
    )


def _bind_action(
    calibrator: RiskCalibrator, penalty: float, boundary: float,
    *, minimum_observations: int | None = None,
    maximum_observations: int | None = None,
    minimum_agreeing_views: int | None = None,
) -> RiskCalibrator:
    return RiskCalibrator(
        benefit_head=calibrator.benefit_head,
        harm_head=calibrator.harm_head,
        risk_penalty=penalty,
        decision_boundary=boundary,
        regions=calibrator.regions,
        evidence_benefit_head=calibrator.evidence_benefit_head,
        evidence_harm_head=calibrator.evidence_harm_head,
        minimum_observations=(
            calibrator.minimum_observations
            if minimum_observations is None else minimum_observations
        ),
        maximum_observations=(
            calibrator.maximum_observations
            if maximum_observations is None else maximum_observations
        ),
        minimum_agreeing_views=(
            calibrator.minimum_agreeing_views
            if minimum_agreeing_views is None else minimum_agreeing_views
        ),
    )


def _constant_logistic_head(probability: float) -> RiskLogisticHead:
    width = len(AggregateEvidenceFeatures.__dataclass_fields__)
    probability = min(1.0 - 1e-12, max(1e-12, float(probability)))
    return RiskLogisticHead(
        intercept=float(np.log(probability / (1.0 - probability))),
        coefficients=(0.0,) * width,
        means=(0.0,) * width,
        scales=(1.0,) * width,
    )


def _fit_logistic_head(
    examples: Sequence[Any],
    *, target: str, feature_mode: str, regularization: float,
    balanced: bool,
) -> RiskLogisticHead:
    if target not in {"benefit", "harm"}:
        raise ValueError("aggregate risk target must be benefit or harm")
    indices = EVIDENCE_FEATURE_MODES[feature_mode]
    width = len(AggregateEvidenceFeatures.__dataclass_fields__)
    if not examples:
        return _constant_logistic_head(0.0)
    matrix = np.asarray([
        [values[index] for index in indices]
        for example in examples
        for values in (_field_values(example.evidence_features),)
    ], dtype=float)
    targets = np.asarray([
        (
            example.correction_units > 0
            if target == "benefit" else example.corruption_units > 0
        )
        for example in examples
    ], dtype=int)
    counts = Counter(example.group for example in examples)
    weights = np.asarray([
        1.0 / counts[example.group] for example in examples
    ], dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0.0] = 1.0
    if len(set(targets)) == 1:
        return _constant_logistic_head(float(targets[0]))
    model = LogisticRegression(
        C=regularization,
        class_weight="balanced" if balanced else None,
        solver="liblinear",
        max_iter=1000,
        random_state=0,
    ).fit((matrix - means) / scales, targets, sample_weight=weights)
    coefficients = np.zeros(width, dtype=float)
    all_means = np.zeros(width, dtype=float)
    all_scales = np.ones(width, dtype=float)
    coefficients[list(indices)] = model.coef_[0]
    all_means[list(indices)] = means
    all_scales[list(indices)] = scales
    return RiskLogisticHead(
        intercept=float(model.intercept_[0]),
        coefficients=tuple(float(value) for value in coefficients),
        means=tuple(float(value) for value in all_means),
        scales=tuple(float(value) for value in all_scales),
    )


def _fit_aggregate_calibrator(
    topics: Sequence[_RiskTopic],
    configuration: AggregateRiskConfiguration,
    *, excluded_groups: frozenset[str] = frozenset(),
) -> RiskCalibrator:
    examples = tuple(
        example
        for topic in topics
        for example in topic.examples
        if example.group not in excluded_groups
    )
    zero = RiskLinearHead((0.0,) * 18)
    common = {
        "feature_mode": configuration.feature_mode,
        "regularization": configuration.regularization,
        "balanced": configuration.balanced,
    }
    return RiskCalibrator(
        zero, zero,
        configuration.risk_penalty,
        configuration.decision_boundary,
        evidence_benefit_head=_fit_logistic_head(
            examples, target="benefit", **common,
        ),
        evidence_harm_head=_fit_logistic_head(
            examples, target="harm", **common,
        ),
        minimum_observations=configuration.minimum_observations,
        maximum_observations=configuration.maximum_observations,
        minimum_agreeing_views=configuration.minimum_agreeing_views,
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
    verifier_evidence: Mapping[
        tuple[str, str, str, int], CandidateFreeVerifierEvidence
    ] | None = None,
    *,
    proposal_verifier_confidence: float = 0.6,
    verifier_proposal_agreement: float = 0.4,
) -> PolicyMetrics:
    cells = Counter({
        f"{topic.record.backbone}/{topic.record.benchmark}": 0
        for topic in topics
    })
    datasets = Counter({topic.record.benchmark: 0 for topic in topics})
    backbones = Counter({topic.record.backbone: 0 for topic in topics})
    corrections = corruptions = observations = selections = 0
    verifier_lookup = _verifier_lookup(topics, verifier_evidence)
    for topic in topics:
        if verifier_lookup is None:
            delta, correction, corruption, selected, observed = (
                _risk_topic_decision(
                    topic, calibrator_for(topic),
                )
            )
        else:
            outcome = candidate_free_cascade_outcome(
                topic, calibrator_for(topic),
                verifier_lookup[verifier_evidence_key(topic.record)],
                proposal_confidence=proposal_verifier_confidence,
                proposal_agreement=verifier_proposal_agreement,
            )
            delta = outcome.net_gain
            correction = outcome.corrections
            corruption = outcome.corruptions
            selected = outcome.selected_source != "P0"
            observed = outcome.observations
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


def _candidate_from_scores(
    topic: _RiskTopic,
    scores: Sequence[tuple[float, float]],
    *,
    penalty: float,
    boundary: float,
    minimum_observations: int,
    maximum_observations: int,
) -> Any:
    """Replay one action rule from cached base-head benefit/harm scores."""
    index = _candidate_index_from_scores(
        topic, scores,
        penalty=penalty,
        boundary=boundary,
        minimum_observations=minimum_observations,
        maximum_observations=maximum_observations,
    )
    return None if index is None else topic.examples[index]


def _candidate_index_from_scores(
    topic: _RiskTopic,
    scores: Sequence[tuple[float, float]],
    *,
    penalty: float,
    boundary: float,
    minimum_observations: int,
    maximum_observations: int,
) -> int | None:
    """Return the selected example index from cached base-head scores."""
    if len(scores) != len(topic.examples):
        raise ValueError("cached aggregate scores do not align with examples")
    checkpoints: list[list[int]] = []
    for index, example in enumerate(topic.examples):
        if (
            not checkpoints
            or topic.examples[checkpoints[-1][0]].checkpoint
            != example.checkpoint
        ):
            checkpoints.append([])
        checkpoints[-1].append(index)
    for indices in checkpoints:
        first = topic.examples[indices[0]]
        if first.observations > maximum_observations:
            break
        selected_index = min(
            indices,
            key=lambda index: (
                -(
                    scores[index][0]
                    - penalty * scores[index][1]
                    - boundary
                ),
                -topic.examples[index].features.agreement,
                -topic.examples[index].features.support,
            ),
        )
        selected = topic.examples[selected_index]
        if (
            selected.observations >= minimum_observations
            and selected.agreeing_views >= MINIMUM_AGREEING_VIEWS
            and scores[selected_index][0]
            - penalty * scores[selected_index][1]
            - boundary >= 0.0
        ):
            return selected_index
    return None


def _metrics_for_cached_verifier_topics(
    topics: Sequence[_RiskTopic],
    scores: Sequence[Sequence[tuple[float, float]]],
    configuration: AggregateRiskConfiguration,
    verifier_evidence: Mapping[
        tuple[str, str, str, int], CandidateFreeVerifierEvidence
    ],
) -> PolicyMetrics:
    """Score a verifier cascade while reusing base-head predictions."""
    if len(scores) != len(topics):
        raise ValueError("cached aggregate topics do not align")
    lookup = _verifier_lookup(topics, verifier_evidence)
    if lookup is None:
        raise ValueError("cached verifier scoring requires evidence")
    cells = Counter({
        f"{topic.record.backbone}/{topic.record.benchmark}": 0
        for topic in topics
    })
    datasets = Counter({topic.record.benchmark: 0 for topic in topics})
    backbones = Counter({topic.record.backbone: 0 for topic in topics})
    corrections = corruptions = observations = selections = 0
    for topic, topic_scores in zip(topics, scores):
        selected = _candidate_from_scores(
            topic, topic_scores,
            penalty=configuration.risk_penalty,
            boundary=configuration.decision_boundary,
            minimum_observations=configuration.minimum_observations,
            maximum_observations=configuration.maximum_observations,
        )
        outcome = candidate_free_outcome_for_selected(
            selected, lookup[verifier_evidence_key(topic.record)],
            proposal_confidence=(
                configuration.proposal_verifier_confidence
                if configuration.proposal_verifier_confidence is not None
                else 0.6
            ),
            proposal_agreement=(
                configuration.verifier_proposal_agreement
                if configuration.verifier_proposal_agreement is not None
                else 0.4
            ),
        )
        cell = f"{topic.record.backbone}/{topic.record.benchmark}"
        cells[cell] += outcome.net_gain
        datasets[topic.record.benchmark] += outcome.net_gain
        backbones[topic.record.backbone] += outcome.net_gain
        corrections += outcome.corrections
        corruptions += outcome.corruptions
        observations += outcome.observations
        selections += outcome.selected_source != "P0"
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


def _verifier_outcome_grid(
    topics: Sequence[_RiskTopic],
    verifier_evidence: Mapping[
        tuple[str, str, str, int], CandidateFreeVerifierEvidence
    ],
    actions: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, int]:
    """Precompute every threshold outcome for each possible selected example."""
    lookup = _verifier_lookup(topics, verifier_evidence)
    if lookup is None:
        raise ValueError("verifier outcome grid requires evidence")
    maximum = max((len(topic.examples) for topic in topics), default=0)
    grid = np.zeros(
        (len(actions), len(topics), maximum + 1, 4), dtype=np.int64,
    )
    for action_index, (confidence, agreement) in enumerate(actions):
        for topic_index, topic in enumerate(topics):
            evidence = lookup[verifier_evidence_key(topic.record)]
            for selected_index in range(len(topic.examples) + 1):
                selected = (
                    None
                    if selected_index == len(topic.examples)
                    else topic.examples[selected_index]
                )
                outcome = candidate_free_outcome_for_selected(
                    selected, evidence,
                    proposal_confidence=confidence,
                    proposal_agreement=agreement,
                )
                target_index = (
                    maximum if selected is None else selected_index
                )
                grid[action_index, topic_index, target_index] = (
                    outcome.corrections,
                    outcome.corruptions,
                    outcome.observations,
                    outcome.selected_source != "P0",
                )
    return grid, maximum


def _metrics_for_cached_verifier_grid(
    topics: Sequence[_RiskTopic],
    scores: Sequence[Sequence[tuple[float, float]]],
    configuration: AggregateRiskConfiguration,
    outcome_grid: np.ndarray,
    no_selection_index: int,
    metric_projection: tuple[
        tuple[str, ...], tuple[str, ...], tuple[str, ...],
        np.ndarray, np.ndarray, np.ndarray,
    ] | None = None,
) -> tuple[PolicyMetrics, ...]:
    """Vectorize the fixed verifier action grid after aggregate selection."""
    selected_indices = np.asarray([
        no_selection_index if index is None else index
        for topic, topic_scores in zip(topics, scores)
        for index in (_candidate_index_from_scores(
            topic, topic_scores,
            penalty=configuration.risk_penalty,
            boundary=configuration.decision_boundary,
            minimum_observations=configuration.minimum_observations,
            maximum_observations=configuration.maximum_observations,
        ),)
    ], dtype=np.int64)
    gathered = outcome_grid[
        :, np.arange(len(topics), dtype=np.int64), selected_indices, :,
    ]
    deltas = gathered[:, :, 0] - gathered[:, :, 1]
    if metric_projection is None:
        metric_projection = _topic_metric_projection(topics)
    (
        cell_names, dataset_names, backbone_names,
        cell_indicator, dataset_indicator, backbone_indicator,
    ) = metric_projection
    cell_deltas = deltas @ cell_indicator
    dataset_deltas = deltas @ dataset_indicator
    backbone_deltas = deltas @ backbone_indicator
    totals = gathered.sum(axis=1)
    return tuple(
        PolicyMetrics(
            net_gain=int(totals[index, 0] - totals[index, 1]),
            corrections=int(totals[index, 0]),
            corruptions=int(totals[index, 1]),
            observations=int(totals[index, 2]),
            selections=int(totals[index, 3]),
            cell_deltas=tuple(
                (name, int(value))
                for name, value in zip(cell_names, cell_deltas[index])
            ),
            dataset_deltas=tuple(
                (name, int(value))
                for name, value in zip(dataset_names, dataset_deltas[index])
            ),
            backbone_deltas=tuple(
                (name, int(value))
                for name, value in zip(backbone_names, backbone_deltas[index])
            ),
        )
        for index in range(outcome_grid.shape[0])
    )


def _topic_metric_projection(
    topics: Sequence[_RiskTopic],
) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...],
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Build fixed topic-to-cell projection matrices once per grid search."""
    cell_names = tuple(sorted({
        f"{topic.record.backbone}/{topic.record.benchmark}"
        for topic in topics
    }))
    dataset_names = tuple(sorted({topic.record.benchmark for topic in topics}))
    backbone_names = tuple(sorted({topic.record.backbone for topic in topics}))

    def indicator(names: Sequence[str], value_for: Callable[[_RiskTopic], str]):
        return np.asarray([
            [int(value_for(topic) == name) for name in names]
            for topic in topics
        ], dtype=np.int64)

    return (
        cell_names, dataset_names, backbone_names,
        indicator(
            cell_names,
            lambda topic: f"{topic.record.backbone}/{topic.record.benchmark}",
        ),
        indicator(dataset_names, lambda topic: topic.record.benchmark),
        indicator(backbone_names, lambda topic: topic.record.backbone),
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
    *,
    verifier_evidence: Mapping[
        tuple[str, str, str, int], CandidateFreeVerifierEvidence
    ] | None = None,
) -> SharedRiskSelection:
    """Select one global aggregate-risk configuration with grouped OOF."""
    frozen = _validate_shared_records(records)
    topics = _risk_topics(frozen)
    verifier_lookup = _verifier_lookup(topics, verifier_evidence)
    groups = tuple(sorted({record.group for record in frozen}))
    splitter = GroupKFold(n_splits=min(4, len(groups)))
    group_folds = tuple(
        tuple(groups[index] for index in held_out)
        for _, held_out in splitter.split(
            np.arange(len(groups)), groups=np.asarray(groups),
        )
    )
    fold_for_group = {
        group: fold_index
        for fold_index, held_out in enumerate(group_folds)
        for group in held_out
    }
    bases = tuple(
        AggregateRiskConfiguration(
            feature_mode, regularization, balanced,
            1.0, 0.0, 1, 14,
        )
        for feature_mode in EVIDENCE_FEATURE_MODES
        for regularization in RISK_L2
        for balanced in EVIDENCE_BALANCING
    )
    cached = {
        (base_index, fold_index): _fit_aggregate_calibrator(
            topics, base, excluded_groups=frozenset(held_out),
        )
        for base_index, base in enumerate(bases)
        for fold_index, held_out in enumerate(group_folds)
    }
    cached_scores = (
        {
            base_index: tuple(
                tuple(
                    cached[(
                        base_index,
                        fold_for_group[topic.record.group],
                    )].predict(
                        example.features, example.evidence_features,
                    )[:2]
                    for example in topic.examples
                )
                for topic in topics
            )
            for base_index in range(len(bases))
        }
        if verifier_lookup is not None else None
    )
    candidates = []
    grid_index = 0
    verifier_actions = (
        tuple(
            (confidence, agreement)
            for confidence in PROPOSAL_VERIFIER_CONFIDENCES
            for agreement in VERIFIER_PROPOSAL_AGREEMENTS
        )
        if verifier_lookup is not None else ((None, None),)
    )
    verifier_outcomes = (
        _verifier_outcome_grid(topics, verifier_lookup, verifier_actions)
        if verifier_lookup is not None else None
    )
    metric_projection = (
        _topic_metric_projection(topics)
        if verifier_lookup is not None else None
    )
    for base_index, base in enumerate(bases):
        for penalty in RISK_PENALTIES:
            for boundary in RISK_BOUNDARIES:
                for minimum in MINIMUM_OBSERVATIONS:
                    for maximum in MAXIMUM_OBSERVATIONS:
                        if minimum > maximum:
                            continue
                        base_configuration = AggregateRiskConfiguration(
                            base.feature_mode, base.regularization,
                            base.balanced, penalty, boundary,
                            minimum, maximum,
                        )
                        if cached_scores is not None:
                            outcome_grid, no_selection = verifier_outcomes
                            action_metrics = _metrics_for_cached_verifier_grid(
                                topics, cached_scores[base_index],
                                base_configuration, outcome_grid, no_selection,
                                metric_projection,
                            )
                        else:
                            fold_calibrators = {
                                fold_index: _bind_action(
                                    cached[(base_index, fold_index)],
                                    penalty, boundary,
                                    minimum_observations=minimum,
                                    maximum_observations=maximum,
                                    minimum_agreeing_views=(
                                        MINIMUM_AGREEING_VIEWS
                                    ),
                                )
                                for fold_index in range(len(group_folds))
                            }
                            action_metrics = (_metrics_for_topics(
                                topics,
                                lambda topic, lookup=fold_calibrators: lookup[
                                    fold_for_group[topic.record.group]
                                ],
                            ),)
                        for (confidence, agreement), oof_metrics in zip(
                            verifier_actions, action_metrics,
                        ):
                            configuration = AggregateRiskConfiguration(
                                base.feature_mode, base.regularization,
                                base.balanced, penalty, boundary,
                                minimum, maximum,
                                proposal_verifier_confidence=confidence,
                                verifier_proposal_agreement=agreement,
                            )
                            candidates.append((
                                robust_rank(
                                    oof_metrics, len(topics), grid_index,
                                    criteria,
                                ),
                                configuration,
                                oof_metrics,
                            ))
                            grid_index += 1
    _, selected, selected_metrics = min(candidates, key=lambda item: item[0])
    refit = _fit_aggregate_calibrator(topics, selected)
    folds = tuple(
        FoldAudit(
            held_out_groups=held_out,
            train_groups=tuple(
                group for group in groups if group not in held_out
            ),
            calibration_samples=sum(
                example.group not in held_out
                for topic in topics
                for example in topic.examples
            ),
        )
        for held_out in group_folds
    )
    return SharedRiskSelection(
        configuration=selected,
        metrics=selected_metrics,
        failures=evaluate_acceptance(
            selected_metrics, len(topics), criteria,
        ),
        folds=folds,
        refit_calibrators=(("*/*", refit),),
        candidate_count=sum(len(topic.examples) for topic in topics),
        source_group_count=len(groups),
        verifier_cascade=verifier_lookup is not None,
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
    *,
    verifier_evidence: Mapping[
        tuple[str, str, str, int], CandidateFreeVerifierEvidence
    ] | None = None,
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
        selected = select_shared_configuration(
            train, _scaled_criteria(criteria, train),
            verifier_evidence=verifier_evidence,
        )
        train_groups = tuple(sorted({record.group for record in train}))
        held_out_topics = _risk_topics(held_out)
        refit = dict(selected.refit_calibrators)
        metrics = _metrics_for_topics(
            held_out_topics,
            lambda topic, lookup=refit: _calibrator_for(lookup, topic),
            verifier_evidence,
            proposal_verifier_confidence=(
                selected.configuration.proposal_verifier_confidence
                if isinstance(
                    selected.configuration, AggregateRiskConfiguration,
                ) and selected.configuration.proposal_verifier_confidence
                is not None else 0.6
            ),
            verifier_proposal_agreement=(
                selected.configuration.verifier_proposal_agreement
                if isinstance(
                    selected.configuration, AggregateRiskConfiguration,
                ) and selected.configuration.verifier_proposal_agreement
                is not None else 0.4
            ),
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
    all_criteria = _scaled_criteria(criteria, all_records)
    failures = evaluate_acceptance(combined, len(all_records), all_criteria)
    refit_selection = None
    refit_policy = None
    if not failures:
        refit_selection = select_shared_configuration(
            all_records, all_criteria, verifier_evidence=verifier_evidence,
        )
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
