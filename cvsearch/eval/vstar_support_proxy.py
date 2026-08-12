"""Evaluator-only geometric evidence-sufficiency proxy for V* calibration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from cvsearch.evidence_gap.types import EXPAND, ZOOM


DETAIL_MIN_NATIVE_RATIO = 0.01


def _crop(value: Any) -> tuple[float, float, float, float]:
    if type(value) not in (list, tuple) or len(value) != 4:
        raise ValueError("crop must contain four coordinates")
    if any(
        isinstance(item, bool) or not isinstance(item, Real)
        or not math.isfinite(float(item)) for item in value
    ):
        raise ValueError("crop coordinates must be finite numbers")
    x1, y1, x2, y2 = (float(item) for item in value)
    if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1:
        raise ValueError("crop must be a positive XYXY rectangle")
    return x1, y1, x2, y2


def _crops(value: Any) -> tuple[tuple[float, float, float, float], ...]:
    if type(value) is not list or not value:
        raise ValueError("crop collection must be a nonempty exact list")
    return tuple(_crop(item) for item in value)


def visible_crops_from_audit(
    action: str, audit: Mapping[str, Any],
) -> tuple[
    tuple[tuple[float, float, float, float], ...],
    tuple[tuple[float, float, float, float], ...],
]:
    """Return native current/candidate crops without reading answer labels."""
    if type(audit) is not dict:
        raise TypeError("action audit must be an exact dictionary")
    if action == ZOOM:
        mappings = audit.get("coordinate_mapping")
        if type(mappings) is not list or not mappings:
            raise ValueError("ZOOM audit requires coordinate mappings")
        if any(type(item) is not dict for item in mappings):
            raise ValueError("ZOOM mappings must be exact dictionaries")
        return (
            tuple(_crop(item.get("current_crop_xyxy")) for item in mappings),
            tuple(_crop(item.get("candidate_crop_xyxy")) for item in mappings),
        )
    if action == EXPAND:
        focus = audit.get("focus_merge_identity")
        context = audit.get("context_merge_identity")
        if type(focus) is not dict or type(context) is not dict:
            raise ValueError("EXPAND audit requires focus and context identities")
        current = _crops(focus.get("merged_crop_xyxy"))
        added = _crops(context.get("merged_crop_xyxy"))
        return current, current + added
    raise ValueError("only ZOOM and EXPAND audits expose this support proxy")


def _target_box(value: Any) -> tuple[float, float, float, float]:
    if type(value) is not list or len(value) != 4:
        raise ValueError("target bbox must be an exact XYWH list")
    if any(
        isinstance(item, bool) or not isinstance(item, Real)
        or not math.isfinite(float(item)) for item in value
    ):
        raise ValueError("target bbox values must be finite numbers")
    x, y, width, height = (float(item) for item in value)
    if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError("target bbox must have positive extent")
    return x, y, width, height


def _visible(
    target: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    *, require_detail: bool,
) -> bool:
    x, y, width, height = target
    x1, y1, x2, y2 = crop
    center_visible = x1 <= x + width / 2 <= x2 and y1 <= y + height / 2 <= y2
    if not center_visible or not require_detail:
        return center_visible
    return min(width / (x2 - x1), height / (y2 - y1)) >= DETAIL_MIN_NATIVE_RATIO


def vstar_geometry_support_label(
    annotation: Mapping[str, Any],
    crops: Sequence[Sequence[Real]],
) -> int:
    """Label spatial coverage and a fixed, answer-independent detail proxy."""
    if type(annotation) is not dict:
        raise TypeError("annotation must be an exact dictionary")
    raw_targets = annotation.get("bbox")
    if type(raw_targets) is not list or not raw_targets:
        raise ValueError("V* annotation requires target boxes")
    targets = tuple(_target_box(value) for value in raw_targets)
    normalized_crops = tuple(_crop(value) for value in crops)
    if not normalized_crops:
        raise ValueError("at least one visible crop is required")
    test_type = annotation.get("test_type")
    if test_type == "relative_position":
        return int(any(
            all(_visible(target, crop, require_detail=False) for target in targets)
            for crop in normalized_crops
        ))
    if test_type == "direct_attributes":
        return int(all(
            any(_visible(target, crop, require_detail=True) for crop in normalized_crops)
            for target in targets
        ))
    raise ValueError("unsupported V* test_type")


def deterministic_calibration_member(input_image: str) -> bool:
    """Assign all observations from one source image to one fixed partition."""
    if type(input_image) is not str:
        raise TypeError("input_image must be an exact string")
    if not input_image:
        raise ValueError("input_image must be nonempty")
    digest = hashlib.sha256(input_image.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 2 == 0


__all__ = [
    "DETAIL_MIN_NATIVE_RATIO", "deterministic_calibration_member",
    "visible_crops_from_audit", "vstar_geometry_support_label",
]
