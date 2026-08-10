"""Pure label-blind P0 dual-view self-consistency selection."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image

from cvsearch.eval.phase7_uncertainty_confirmation import (
    CONFIRMATION_THRESHOLDS,
    _confidence,
    _exact_snapshot,
    _exact_rgb_image,
    _image_sha256,
    _p0,
    _sha256,
    _snapshot_json,
    _vstar_confirmation_record,
)


P0_CONSISTENCY_THRESHOLDS = CONFIRMATION_THRESHOLDS
_VIEW_FIELDS = frozenset({
    "feasible",
    "output",
    "stability",
    "majority_fraction",
    "view_sha256",
    "prompt_sha256",
})


@dataclass(frozen=True)
class P0SelfConsistencyDecision:
    action: str
    status: str
    output: Any
    confidence: float | None


@dataclass(frozen=True)
class _ReplayState:
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class _ReplayNode:
    state: _ReplayState
    is_root: bool = False
    search_source: str = "fine"


def _validated_boxes(
    value: Any, source: Image.Image,
) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("P0 replay requires nonempty final boxes")
    boxes = []
    for index, raw in enumerate(value):
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f"P0 final boxes[{index}] must contain four values")
        numbers = []
        for item in raw:
            if (
                isinstance(item, bool) or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
            ):
                raise TypeError(f"P0 final boxes[{index}] must be finite numbers")
            numbers.append(float(item))
        x, y, width, height = numbers
        if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
            raise ValueError(f"P0 final boxes[{index}] has invalid geometry")
        if x + width > source.width or y + height > source.height:
            raise ValueError(f"P0 final boxes[{index}] exceeds the source image")
        boxes.append((x, y, width, height))
    return tuple(boxes)


def render_p0_native_views(
    source: Image.Image, final_boxes: Any, model: Any,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Replay final P0 boxes through the native fine-node renderer."""
    source = _exact_rgb_image(source, "P0 source image")
    source_sha256 = _image_sha256(source)
    boxes = _validated_boxes(final_boxes, source)
    renderer = getattr(model, "process_nodes_to_image_list", None)
    if not callable(renderer):
        raise TypeError("P0 replay model must expose the native node renderer")
    nodes = tuple(_ReplayNode(_ReplayState(box)) for box in boxes)
    views = renderer(nodes, source.copy(), root_anyres=True)
    if not isinstance(views, (list, tuple)) or len(views) != 3:
        raise ValueError("P0 native renderer must return context, crop, and focus views")
    context = _exact_rgb_image(views[0], "P0 context view")
    crop = _exact_rgb_image(views[1], "P0 crop view")
    focus = _exact_rgb_image(views[2], "P0 focus view")
    focus_sha256 = _image_sha256(focus)
    context_sha256 = _image_sha256(context)
    if focus_sha256 == context_sha256:
        raise ValueError("P0 focus and context views must be pixel-distinct")
    return focus, context, {
        "renderer": "native-fine-node-v1",
        "boxes": [list(box) for box in boxes],
        "source_image_sha256": source_sha256,
        "focus_view_sha256": focus_sha256,
        "context_view_sha256": context_sha256,
        "crop_view_sha256": _image_sha256(crop),
        "focus_size": [focus.width, focus.height],
        "context_size": [context.width, context.height],
        "crop_size": [crop.width, crop.height],
    }


def aggregate_p0_view(
    options: Sequence[str], observations: Any,
) -> dict[str, Any]:
    """Aggregate three loss rows and require their majority to match the mean."""
    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise TypeError("P0 self-consistency options must be a sequence")
    frozen_options = tuple(options)
    if not frozen_options or not all(
        isinstance(option, str) and option for option in frozen_options
    ):
        raise ValueError("P0 self-consistency options must be nonempty strings")
    record = _vstar_confirmation_record(
        frozen_options, observations, "P0 self-consistency view",
    )
    winners = [item["winner"] for item in observations]
    counts = Counter(winners)
    majority_output, majority_count = min(
        counts.items(), key=lambda item: (-item[1], item[0]),
    )
    majority_fraction = majority_count / len(winners)
    feasible = (
        majority_count >= 2
        and majority_output == record.output
        and record.aggregation_available is not False
    )
    confidence = min(float(record.confidence), majority_fraction)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("P0 self-consistency confidence must be finite in [0, 1]")
    return {
        "feasible": feasible,
        "output": record.output if feasible else None,
        "confidence": confidence,
        "majority_fraction": majority_fraction,
        "aggregate_output": record.output,
        "majority_output": majority_output,
    }


def _view(value: Any, name: str) -> tuple[dict[str, Any], float | None]:
    view = _exact_snapshot(value, _VIEW_FIELDS, name)
    if type(view["feasible"]) is not bool:
        raise TypeError(f"{name} feasible must be an exact bool")
    prompts = view["prompt_sha256"]
    if type(prompts) is not list or len(prompts) != 3:
        raise ValueError(f"{name} must bind exactly three prompt hashes")
    for index, prompt_hash in enumerate(prompts):
        _sha256(prompt_hash, f"{name} prompt hash {index}")
    if len(set(prompts)) != 3:
        raise ValueError(f"{name} prompt hashes must be distinct")
    if not view["feasible"]:
        if view["view_sha256"] is not None:
            _sha256(view["view_sha256"], f"{name} view hash")
        if any(
            view[field] is not None
            for field in ("output", "stability", "majority_fraction")
        ):
            raise ValueError(f"infeasible {name} must not expose a projection")
        return view, None
    _sha256(view["view_sha256"], f"{name} view hash")
    if view["output"] is None:
        raise ValueError(f"feasible {name} must expose one output")
    confidence = _confidence(view["stability"], f"{name} stability")
    majority = view["majority_fraction"]
    if (
        type(majority) not in (int, float)
        or not math.isfinite(float(majority))
        or float(majority) not in (2 / 3, 1.0)
    ):
        raise ValueError(f"{name} must have a three-prompt strict majority")
    return view, confidence


def _retained(p0: dict[str, Any]) -> P0SelfConsistencyDecision:
    return P0SelfConsistencyDecision(
        action="P0",
        status="retained_p0",
        output=_snapshot_json(p0["output"]),
        confidence=None,
    )


def select_p0_self_consistency(
    p0: dict[str, Any], focus: dict[str, Any], context: dict[str, Any],
    *, threshold: float,
) -> P0SelfConsistencyDecision:
    """Admit one alternative only after two three-prompt views agree."""
    if (
        type(threshold) not in (int, float)
        or float(threshold) not in P0_CONSISTENCY_THRESHOLDS
    ):
        raise ValueError("P0 consistency threshold must be one frozen coarse value")
    threshold = float(threshold)
    p0, _ = _p0(p0)
    focus, focus_confidence = _view(focus, "focus view")
    context, context_confidence = _view(context, "context view")
    if not focus["feasible"] or not context["feasible"]:
        return _retained(p0)
    if focus["view_sha256"] == context["view_sha256"]:
        return _retained(p0)
    if focus["prompt_sha256"] != context["prompt_sha256"]:
        return _retained(p0)
    if focus["output"] != context["output"]:
        return _retained(p0)
    if focus["output"] == p0["output"]:
        return _retained(p0)
    if focus_confidence is None or context_confidence is None:
        raise AssertionError("feasible P0 consistency view lost its confidence")
    confidence = min(focus_confidence, context_confidence)
    if confidence < threshold:
        return _retained(p0)
    return P0SelfConsistencyDecision(
        action="CONSISTENCY",
        status="selected_p0_self_consistency",
        output=_snapshot_json(focus["output"]),
        confidence=confidence,
    )
