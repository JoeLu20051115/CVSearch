#!/usr/bin/env python3
"""Auditable acceptance gates for robust Stage-3 policy transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .freeze_uncertainty_support import BENCHMARKS, PolicyMetrics


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
