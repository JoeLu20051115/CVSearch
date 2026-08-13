#!/usr/bin/env python3
"""Label-blind replay primitives for the uncertainty-support selector."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass
from typing import Any


PROFILES: Mapping[str, tuple[float, ...]] = {
    "balanced": (0.20, 0.20, 0.20, 0.20, 0.20),
    "support_heavy": (0.10, 0.15, 0.35, 0.20, 0.20),
    "uncertainty_light": (0.10, 0.25, 0.25, 0.20, 0.20),
}
THRESHOLDS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)


def _unit_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class AdvantageFeatures:
    """The five normalized candidate-vs-P0 inputs to the unified score."""

    uncertainty: float
    agreement: float
    support: float
    support_gain_01: float
    conflict_margin_01: float

    def __post_init__(self) -> None:
        for name, value in zip(self.__dataclass_fields__, astuple(self)):
            object.__setattr__(self, name, _unit_float(value, name))


def raw_advantage(
    features: AdvantageFeatures,
    weights: Sequence[float],
) -> float:
    """Return the sole weighted numeric score used by the selector."""
    if not isinstance(features, AdvantageFeatures):
        raise TypeError("features must be AdvantageFeatures")
    if len(weights) != 5:
        raise ValueError("weight profile must contain five values")
    normalized = tuple(_unit_float(value, "weight") for value in weights)
    if not math.isclose(math.fsum(normalized), 1.0, abs_tol=1e-12):
        raise ValueError("weight profile must sum to one")
    return math.fsum(
        weight * value for weight, value in zip(normalized, astuple(features))
    )


@dataclass(frozen=True)
class UtilityIsotonicCalibrator:
    """A deterministic monotone mapping from raw score to relative utility."""

    upper_bounds: tuple[float, ...]
    utilities: tuple[float, ...]

    def _validate(self) -> None:
        if not self.upper_bounds or len(self.upper_bounds) != len(self.utilities):
            raise ValueError("utility calibrator arrays must be nonempty and aligned")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.upper_bounds
        ):
            raise ValueError("utility calibrator bounds must be finite")
        if any(
            left >= right
            for left, right in zip(self.upper_bounds, self.upper_bounds[1:])
        ):
            raise ValueError("utility calibrator bounds must be strictly increasing")
        values = tuple(_unit_float(value, "utility") for value in self.utilities)
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("utility calibrator values must be nondecreasing")

    def predict(self, score: float) -> float:
        self._validate()
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("utility score must be numeric")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("utility score must be finite")
        index = min(bisect_left(self.upper_bounds, value), len(self.utilities) - 1)
        return float(self.utilities[index])

    def to_dict(self) -> dict[str, list[float]]:
        self._validate()
        return {
            "upper_bounds": [float(value) for value in self.upper_bounds],
            "utilities": [float(value) for value in self.utilities],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UtilityIsotonicCalibrator:
        if not isinstance(payload, Mapping):
            raise TypeError("utility calibrator payload must be a mapping")
        bounds = payload.get("upper_bounds")
        utilities = payload.get("utilities")
        if not isinstance(bounds, list) or not isinstance(utilities, list):
            raise ValueError("utility calibrator payload arrays are missing")
        result = cls(tuple(bounds), tuple(utilities))
        result._validate()
        return result


def fit_utility_isotonic(
    samples: Sequence[tuple[float, float]],
) -> UtilityIsotonicCalibrator:
    """Fit continuous-target isotonic regression with deterministic PAVA."""
    if not samples:
        raise ValueError("utility calibration samples must be nonempty")
    validated = []
    for score, target in samples:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("utility calibration score must be numeric")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("utility calibration score must be finite")
        validated.append((score, _unit_float(target, "utility target")))
    validated.sort()

    grouped: list[tuple[float, float, int]] = []
    for score, target in validated:
        if grouped and grouped[-1][0] == score:
            bound, total, count = grouped[-1]
            grouped[-1] = (bound, total + target, count + 1)
        else:
            grouped.append((score, target, 1))

    blocks: list[dict[str, float | int]] = []
    for index, (_, total, count) in enumerate(grouped):
        blocks.append({"start": index, "end": index, "total": total, "count": count})
        while len(blocks) >= 2:
            left, right = blocks[-2:]
            left_mean = float(left["total"]) / int(left["count"])
            right_mean = float(right["total"]) / int(right["count"])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [{
                "start": int(left["start"]),
                "end": int(right["end"]),
                "total": float(left["total"]) + float(right["total"]),
                "count": int(left["count"]) + int(right["count"]),
            }]

    utilities = [0.0] * len(grouped)
    for block in blocks:
        mean = float(block["total"]) / int(block["count"])
        for index in range(int(block["start"]), int(block["end"]) + 1):
            utilities[index] = mean
    result = UtilityIsotonicCalibrator(
        tuple(item[0] for item in grouped),
        tuple(utilities),
    )
    result._validate()
    return result
