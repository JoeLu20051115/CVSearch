"""Evaluator-only geometric evidence-sufficiency proxy for TreeBench."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


DETAIL_MIN_NATIVE_RATIO = 0.01


def _box(value: Any, name: str) -> tuple[float, float, float, float]:
    if (
        type(value) not in (list, tuple) or len(value) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, Real)
            or not math.isfinite(float(item)) for item in value
        )
    ):
        raise ValueError(f"{name} must contain four finite coordinates")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must be a positive XYXY box")
    return x1, y1, x2, y2


def _targets(annotation: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
    raw = annotation.get("target_instances")
    if type(raw) is not str:
        raise TypeError("TreeBench target_instances must be a JSON string")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("TreeBench target_instances is not valid JSON") from error
    if type(values) is not list or not values:
        raise ValueError("TreeBench target_instances must be a nonempty list")
    return tuple(_box(value, "target instance") for value in values)


def _center_visible(
    target: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> bool:
    tx1, ty1, tx2, ty2 = target
    cx1, cy1, cx2, cy2 = crop
    return cx1 <= (tx1 + tx2) / 2 <= cx2 and cy1 <= (ty1 + ty2) / 2 <= cy2


def _detailed(
    target: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> bool:
    if not _center_visible(target, crop):
        return False
    tx1, ty1, tx2, ty2 = target
    cx1, cy1, cx2, cy2 = crop
    return min((tx2 - tx1) / (cx2 - cx1), (ty2 - ty1) / (cy2 - cy1)) >= (
        DETAIL_MIN_NATIVE_RATIO
    )


def treebench_geometry_support_label(
    annotation: Mapping[str, Any], crops: Sequence[Sequence[Real]],
) -> int:
    """Label post-hoc target visibility without exposing boxes to inference."""
    if type(annotation) is not dict:
        raise TypeError("TreeBench annotation must be an exact dictionary")
    category = annotation.get("category")
    if type(category) is not str:
        raise TypeError("TreeBench category must be a string")
    targets = _targets(annotation)
    normalized_crops = tuple(_box(crop, "visible crop") for crop in crops)
    if not normalized_crops:
        raise ValueError("at least one visible crop is required")
    if category.startswith("Perception/"):
        return int(all(
            any(_detailed(target, crop) for crop in normalized_crops)
            for target in targets
        ))
    if category.startswith("Reasoning/"):
        return int(any(
            all(_center_visible(target, crop) for target in targets)
            for crop in normalized_crops
        ))
    raise ValueError("unsupported TreeBench category")


__all__ = ["DETAIL_MIN_NATIVE_RATIO", "treebench_geometry_support_label"]
