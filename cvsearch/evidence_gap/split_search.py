"""Bounded, answer-free primitives for Stage 3 split candidate search."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence


SPLIT_GRID = 2
SPLIT_OVERLAP_FRACTION = 0.125
SPLIT_RELEVANCE_WEIGHT = 0.70
SPLIT_VISUAL_FEATURE_WEIGHT = 0.50
SPLIT_MAX_DEPTH = 2
_SHA256 = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)


@dataclass(frozen=True)
class SplitPatch:
    path: tuple[int, ...]
    box: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if type(self.path) is not tuple or len(self.path) > SPLIT_MAX_DEPTH:
            raise ValueError("split path must be a tuple within the depth budget")
        if any(type(index) is not int or not 0 <= index < SPLIT_GRID ** 2
               for index in self.path):
            raise ValueError("split path contains an invalid child index")
        if type(self.box) is not tuple or len(self.box) != 4:
            raise TypeError("split box must be an integer XYXY tuple")
        if any(type(value) is not int for value in self.box):
            raise TypeError("split box must contain exact integers")
        x0, y0, x1, y1 = self.box
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("split box is invalid")

    @property
    def identity(self) -> str:
        return "root" if not self.path else "p" + ".".join(map(str, self.path))


@dataclass(frozen=True)
class RankedSplitPatch:
    patch: SplitPatch
    relevance: float
    edge_density: float
    feature_deviation: float
    relevance_percentile: float
    edge_percentile: float
    feature_deviation_percentile: float
    visual_information: float
    score: float


@dataclass(frozen=True)
class SplitConfirmation:
    confirmed: bool
    canonical_answer: Any
    reason: str
    trajectory_s: int | None
    support_gain: float | None
    selection_score: float | None


def generate_split_children(parent: SplitPatch) -> tuple[SplitPatch, ...]:
    """Return the four fixed-overlap native-coordinate children of ``parent``."""
    if type(parent) is not SplitPatch:
        raise TypeError("split parent must be an exact SplitPatch")
    if len(parent.path) >= SPLIT_MAX_DEPTH:
        raise ValueError("split parent reached the maximum search depth")
    x0, y0, x1, y1 = parent.box
    width, height = x1 - x0, y1 - y0
    children = []
    for row in range(SPLIT_GRID):
        base_y0 = y0 + math.floor(row * height / SPLIT_GRID)
        base_y1 = y0 + math.ceil((row + 1) * height / SPLIT_GRID)
        pad_y = (base_y1 - base_y0) * SPLIT_OVERLAP_FRACTION
        child_y0 = max(y0, math.floor(base_y0 - pad_y))
        child_y1 = min(y1, math.ceil(base_y1 + pad_y))
        for column in range(SPLIT_GRID):
            base_x0 = x0 + math.floor(column * width / SPLIT_GRID)
            base_x1 = x0 + math.ceil((column + 1) * width / SPLIT_GRID)
            pad_x = (base_x1 - base_x0) * SPLIT_OVERLAP_FRACTION
            child_x0 = max(x0, math.floor(base_x0 - pad_x))
            child_x1 = min(x1, math.ceil(base_x1 + pad_x))
            index = row * SPLIT_GRID + column
            children.append(SplitPatch(
                parent.path + (index,),
                (child_x0, child_y0, child_x1, child_y1),
            ))
    if len({child.box for child in children}) != SPLIT_GRID ** 2:
        raise ValueError("split produced duplicate child boxes")
    return tuple(children)


def _finite_values(values: Any, count: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must align with every split patch")
    result = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(float(value))
    return tuple(result)


def _percentiles(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) == 1:
        return (0.5,)
    denominator = len(values) - 1
    return tuple(
        (
            sum(other < value for other in values)
            + (sum(other == value for other in values) - 1) / 2
        ) / denominator
        for value in values
    )


def rank_split_children(
    patches: Sequence[SplitPatch],
    relevance: Any,
    edge_density: Any,
    feature_deviation: Any,
) -> tuple[RankedSplitPatch, ...]:
    """Rank siblings using query relevance and Eq. 4-style visual information."""
    if not isinstance(patches, (list, tuple)) or not patches:
        raise ValueError("split ranking requires nonempty siblings")
    frozen_patches = tuple(patches)
    if not all(type(patch) is SplitPatch for patch in frozen_patches):
        raise TypeError("split ranking patches must be exact SplitPatch values")
    parent_paths = {patch.path[:-1] for patch in frozen_patches}
    if len(parent_paths) != 1 or len({patch.path for patch in frozen_patches}) != len(
        frozen_patches
    ):
        raise ValueError("split ranking requires unique siblings")
    count = len(frozen_patches)
    relevance = _finite_values(relevance, count, "split relevance")
    edge_density = _finite_values(edge_density, count, "split edge density")
    feature_deviation = _finite_values(
        feature_deviation, count, "split feature deviation",
    )
    relevance_percentiles = _percentiles(relevance)
    edge_percentiles = _percentiles(edge_density)
    deviation_percentiles = _percentiles(feature_deviation)
    ranked = []
    for index, patch in enumerate(frozen_patches):
        visual_information = (
            SPLIT_VISUAL_FEATURE_WEIGHT * deviation_percentiles[index]
            + (1 - SPLIT_VISUAL_FEATURE_WEIGHT) * edge_percentiles[index]
        )
        score = (
            SPLIT_RELEVANCE_WEIGHT * relevance_percentiles[index]
            + (1 - SPLIT_RELEVANCE_WEIGHT) * visual_information
        )
        ranked.append(RankedSplitPatch(
            patch=patch,
            relevance=relevance[index],
            edge_density=edge_density[index],
            feature_deviation=feature_deviation[index],
            relevance_percentile=relevance_percentiles[index],
            edge_percentile=edge_percentiles[index],
            feature_deviation_percentile=deviation_percentiles[index],
            visual_information=visual_information,
            score=score,
        ))
    ranked.sort(key=lambda item: (
        -item.score,
        -item.relevance_percentile,
        item.patch.path,
    ))
    return tuple(ranked)


def mann_kendall_s(values: Sequence[float]) -> int:
    """Return the descriptive Mann-Kendall S direction statistic."""
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("support trajectory requires at least two values")
    frozen = _finite_values(values, len(values), "support trajectory")
    return sum(
        (later > earlier) - (later < earlier)
        for index, earlier in enumerate(frozen[:-1])
        for later in frozen[index + 1:]
    )


def _probability(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ValueError(f"{name} must be finite in [0, 1]")
    return float(value)


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _answer(value: Any) -> Any:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.strip():
        return " ".join(value.split())
    if isinstance(value, (list, tuple)) and value:
        result = []
        for component in value:
            if component is None:
                result.append(None)
            elif type(component) is int:
                result.append(component)
            elif isinstance(component, str) and component.strip():
                result.append(" ".join(component.split()))
            else:
                raise ValueError("canonical answer has an invalid component")
        return tuple(result)
    raise ValueError("canonical answer is invalid")


def _project_two_view_answer(p0_answer: Any, tight_answer: Any, context_answer: Any):
    p0 = _answer(p0_answer)
    try:
        tight = _answer(tight_answer)
        context = _answer(context_answer)
    except ValueError:
        return p0, "invalid_answer"
    tuple_inputs = tuple(isinstance(value, tuple) for value in (p0, tight, context))
    if any(tuple_inputs):
        if not all(tuple_inputs) or not len(p0) == len(tight) == len(context):
            return p0, "view_answer_disagreement"
        projected = tuple(
            tight_value
            if tight_value is not None and tight_value == context_value
            else p0_value
            for p0_value, tight_value, context_value in zip(p0, tight, context)
        )
        return projected, None
    if tight != context:
        return p0, "view_answer_disagreement"
    return tight, None


def confirm_split_branch(
    *,
    p0_answer: Any,
    p0_support: float,
    tight_answer: Any,
    tight_support: float,
    tight_view_sha256: str,
    context_answer: Any,
    context_support: float,
    context_view_sha256: str,
    minimum_final_support: float,
    minimum_support_gain: float,
    conflict_answer: Any = None,
    conflict_selection_score: float | None = None,
) -> SplitConfirmation:
    """Confirm one branch only when two distinct views support a safe change."""
    p0 = _answer(p0_answer)
    p0_support = _probability(p0_support, "P0 support")
    tight_support = _probability(tight_support, "tight support")
    context_support = _probability(context_support, "context support")
    minimum_final_support = _probability(
        minimum_final_support, "minimum final support",
    )
    minimum_support_gain = _probability(
        minimum_support_gain, "minimum support gain",
    )
    tight_hash = _hash(tight_view_sha256, "tight view hash")
    context_hash = _hash(context_view_sha256, "context view hash")
    if (conflict_answer is None) != (conflict_selection_score is None):
        raise ValueError("conflict answer and score must be supplied together")
    conflict = None
    if conflict_answer is not None:
        conflict = _answer(conflict_answer)
        conflict_selection_score = _probability(
            conflict_selection_score, "conflict selection score",
        )

    trajectory_s = mann_kendall_s((p0_support, tight_support, context_support))
    selection_score = min(tight_support, context_support)
    support_gain = selection_score - p0_support

    def rejected(reason: str) -> SplitConfirmation:
        return SplitConfirmation(
            False, p0, reason, trajectory_s, support_gain, selection_score,
        )

    projected, projection_failure = _project_two_view_answer(
        p0, tight_answer, context_answer,
    )
    if projection_failure is not None:
        return rejected(projection_failure)
    if tight_hash == context_hash:
        return rejected("duplicate_render")
    if projected == p0:
        return rejected("no_answer_change")
    if trajectory_s < 0:
        return rejected("decreasing_trajectory")
    if selection_score < minimum_final_support:
        return rejected("insufficient_support")
    if support_gain <= minimum_support_gain:
        return rejected("insufficient_support_gain")
    if (
        conflict is not None
        and conflict == p0
        and conflict_selection_score >= selection_score
    ):
        return rejected("equally_strong_p0_conflict")
    return SplitConfirmation(
        True,
        projected,
        "confirmed_two_view_trajectory",
        trajectory_s,
        support_gain,
        selection_score,
    )
