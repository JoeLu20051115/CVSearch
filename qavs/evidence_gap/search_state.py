"""Deterministic, answer-free collection of visual search candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

from PIL import Image

from qavs.models.tree import NodeA, NodeState


class NextNoOpReason(str, Enum):
    EMPTY_QUEUE = "next_queue_empty"
    NO_VALID_CANDIDATES = "next_no_valid_candidates"
    ALL_CANDIDATES_VISITED = "next_all_candidates_visited"
    ALL_OBSERVATIONS_VISITED = "next_all_observations_visited"


class ExpandNoOpReason(str, Enum):
    EMPTY_FOCUS = "expand_p0_focus_empty"
    UNBOUND_FOCUS = "expand_p0_focus_unavailable"
    NONLOCAL_FOCUS = "expand_p0_focus_nonlocal"
    DUPLICATE_CONTEXT = "expand_duplicate_context"
    ADDS_NO_NEW_AREA = "expand_context_adds_no_new_area"
    NO_SPATIAL_CONTEXT = "expand_no_spatially_eligible_unvisited_context"


_NATIVE_SOURCES = frozenset({None, "global", "fast", "fine", "fine_fallback"})


def _renderer_kind(source: str | None) -> str:
    if source == "global":
        return "root"
    if source == "fast":
        return "fast"
    return "fine"


def _renderer_identity(
    source_image_key: str,
    bbox: tuple[int | float, int | float, int | float, int | float],
    render_level: int,
    renderer_kind: str,
) -> str:
    payload = {
        "source_image_key": source_image_key,
        "renderer_kind": renderer_kind,
    }
    if renderer_kind != "root":
        payload.update({"bbox": list(bbox), "render_level": render_level})
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class _RenderSource:
    mode: str
    size: tuple[int, int]
    pixels: bytes = field(repr=False)
    palette: tuple[int, ...] | None = field(repr=False)

    @classmethod
    def capture(cls, image: Image.Image) -> _RenderSource:
        palette = image.getpalette()
        return cls(
            mode=image.mode,
            size=(image.width, image.height),
            pixels=image.tobytes(),
            palette=None if palette is None else tuple(palette),
        )

    def render_image(self) -> Image.Image:
        image = Image.frombytes(self.mode, self.size, self.pixels)
        if self.palette is not None:
            image.putpalette(list(self.palette))
        return image


@dataclass(frozen=True)
class NextCandidate:
    canonical_key: str
    bbox_original: tuple[int | float, int | float, int | float, int | float]
    depth: int
    render_level: int
    posterior_score: float | None
    first_seen_ordinal: int
    tree_scope: str
    crop_origin: tuple[int | float, int | float]
    source_image_key: str
    source: str | None
    renderer_kind: str
    renderer_identity: str
    _render_source: _RenderSource = field(repr=False, compare=False)

    @property
    def render_node(self) -> NodeA:
        """Build a fresh original-image adapter without copying tree state."""
        node = NodeA(NodeState(
            original_image_pil=self._render_source.render_image(),
            bbox=list(self.bbox_original),
        ))
        if self.source is not None:
            node.search_source = self.source
        if self.renderer_kind == "root":
            node.is_root = True
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "bbox_original": list(self.bbox_original),
            "depth": self.depth,
            "render_level": self.render_level,
            "posterior_score": self.posterior_score,
            "first_seen_ordinal": self.first_seen_ordinal,
            "tree_scope": self.tree_scope,
            "crop_origin": list(self.crop_origin),
            "source_image_key": self.source_image_key,
            "source": self.source,
            "renderer_kind": self.renderer_kind,
            "renderer_identity": self.renderer_identity,
        }


@dataclass(frozen=True)
class NextDecision:
    candidate: NextCandidate | None
    no_op_reason: NextNoOpReason | None

    def __post_init__(self) -> None:
        if (self.candidate is None) == (self.no_op_reason is None):
            raise ValueError("NEXT decision must contain exactly one candidate or no-op reason")


@dataclass(frozen=True)
class ExpandDecision:
    """Pure, label-free choice of one native spatial context descriptor."""

    candidate: NextCandidate | None
    no_op_reason: ExpandNoOpReason | None
    current_keys: tuple[str, ...]
    focus_descriptors: tuple[NextCandidate, ...] = ()
    focus_union_xyxy: tuple[float, float, float, float] | None = None
    positive_outside_area: float | None = None
    normalized_edge_gap: float | None = None
    rank_tuple: tuple[float, bool, float, int, str] | None = None

    def __post_init__(self) -> None:
        if (self.candidate is None) == (self.no_op_reason is None):
            raise ValueError("EXPAND decision must contain exactly one candidate or no-op reason")
        if not isinstance(self.current_keys, tuple) or not all(
            isinstance(key, str) and key for key in self.current_keys
        ):
            raise ValueError("EXPAND current_keys must be an immutable string tuple")
        if not isinstance(self.focus_descriptors, tuple) or not all(
            type(item) is NextCandidate for item in self.focus_descriptors
        ):
            raise TypeError("EXPAND focus descriptors must be an immutable native tuple")
        measurements = (
            self.focus_union_xyxy, self.positive_outside_area,
            self.normalized_edge_gap, self.rank_tuple,
        )
        if self.candidate is None and any(value is not None for value in measurements):
            raise ValueError("EXPAND no-op cannot retain candidate measurements")
        if self.candidate is not None and any(value is None for value in measurements):
            raise ValueError("EXPAND candidate requires complete spatial measurements")
        if self.focus_descriptors:
            if tuple(item.canonical_key for item in self.focus_descriptors) != self.current_keys:
                raise ValueError("EXPAND focus descriptors differ from current keys")
            for descriptor in self.focus_descriptors:
                self._validate_descriptor(descriptor, "focus")
            if len({item.renderer_identity for item in self.focus_descriptors}) != len(
                self.focus_descriptors
            ):
                raise ValueError("EXPAND focus renderer identities must be unique")
        if self.candidate is not None:
            if not self.focus_descriptors:
                raise ValueError("EXPAND candidate requires exact focus descriptors")
            self._validate_descriptor(self.candidate, "context")
            source_key = self.focus_descriptors[0].source_image_key
            if (
                any(item.source_image_key != source_key for item in self.focus_descriptors)
                or self.candidate.source_image_key != source_key
                or self.candidate.canonical_key in self.current_keys
                or self.candidate.renderer_identity in {
                    item.renderer_identity for item in self.focus_descriptors
                }
            ):
                raise ValueError("EXPAND selection source/context identity is invalid")
            focus_rectangles = tuple(_xyxy(item) for item in self.focus_descriptors)
            expected_union = (
                min(item[0] for item in focus_rectangles),
                min(item[1] for item in focus_rectangles),
                max(item[2] for item in focus_rectangles),
                max(item[3] for item in focus_rectangles),
            )
            rectangle = _xyxy(self.candidate)
            area = (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])
            expected_outside = area - _intersection_union_area(rectangle, focus_rectangles)
            contained = any(
                rectangle[0] >= item[0] and rectangle[1] >= item[1]
                and rectangle[2] <= item[2] and rectangle[3] <= item[3]
                for item in focus_rectangles
            )
            contains_all = all(
                rectangle[0] <= item[0] and rectangle[1] <= item[1]
                and rectangle[2] >= item[2] and rectangle[3] >= item[3]
                for item in focus_rectangles
            )
            diagonal = math.hypot(*self.candidate._render_source.size)
            expected_gap = min(
                math.hypot(
                    max(item[0] - rectangle[2], rectangle[0] - item[2], 0.0),
                    max(item[1] - rectangle[3], rectangle[1] - item[3], 0.0),
                )
                for item in focus_rectangles
            ) / diagonal
            expected_rank = (
                expected_gap,
                self.candidate.posterior_score is None,
                0.0 if self.candidate.posterior_score is None
                else -self.candidate.posterior_score,
                self.candidate.first_seen_ordinal,
                self.candidate.canonical_key,
            )
            if (
                self.focus_union_xyxy != expected_union
                or self.positive_outside_area != expected_outside
                or self.normalized_edge_gap != expected_gap
                or self.rank_tuple != expected_rank
                or expected_outside <= 0.0 or contained or contains_all
            ):
                raise ValueError("EXPAND selection geometry/rank is not canonical")

    @staticmethod
    def _validate_descriptor(descriptor: NextCandidate, role: str) -> None:
        if (
            descriptor.canonical_key != _canonical_key(
                descriptor.bbox_original, descriptor.depth, descriptor.render_level,
            )
            or descriptor.renderer_identity != _renderer_identity(
                descriptor.source_image_key, descriptor.bbox_original,
                descriptor.render_level, descriptor.renderer_kind,
            )
            or descriptor.depth < 0 or descriptor.render_level < 0
            or descriptor.first_seen_ordinal < 0
            or descriptor.source not in {None, "fast", "fine", "fine_fallback"}
            or descriptor.renderer_kind != (
                "fast" if descriptor.source == "fast" else "fine"
            )
        ):
            raise ValueError(f"EXPAND {role} descriptor identity is not canonical")
        x, y, width, height = (float(value) for value in descriptor.bbox_original)
        source_width, source_height = descriptor._render_source.size
        if (
            not all(math.isfinite(value) for value in (x, y, width, height))
            or x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > source_width or y + height > source_height
            or (
                descriptor.posterior_score is not None
                and not math.isfinite(descriptor.posterior_score)
            )
        ):
            raise ValueError(f"EXPAND {role} descriptor geometry is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "no_op_reason": None if self.no_op_reason is None else self.no_op_reason.value,
            "current_keys": list(self.current_keys),
            "focus_descriptors": [item.to_dict() for item in self.focus_descriptors],
            "focus_union_xyxy": (
                None if self.focus_union_xyxy is None else list(self.focus_union_xyxy)
            ),
            "positive_outside_area": self.positive_outside_area,
            "normalized_edge_gap": self.normalized_edge_gap,
            "rank_tuple": None if self.rank_tuple is None else list(self.rank_tuple),
        }


def _source_identity(image: Image.Image) -> dict[str, Any]:
    return {
        "mode": image.mode,
        "size": [image.width, image.height],
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _source_key(identity: Mapping[str, Any]) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _bbox(value: Any) -> tuple[int | float, int | float, int | float, int | float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("candidate bbox must contain four finite numbers")
    result = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
            raise ValueError("candidate bbox must contain four finite numbers")
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError("candidate bbox must contain four finite numbers")
        result.append(int(number) if number.is_integer() else number)
    return tuple(result)  # type: ignore[return-value]


def _canonical_key(bbox: tuple[Any, ...], depth: int, render_level: int) -> str:
    return json.dumps(
        {"bbox": list(bbox), "depth": depth, "render_level": render_level},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _xyxy(candidate: NextCandidate) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in candidate.bbox_original)
    return x, y, x + width, y + height


def _rect_union_area(rectangles: Sequence[tuple[float, float, float, float]]) -> float:
    if not rectangles:
        return 0.0
    xs = sorted({coordinate for rectangle in rectangles for coordinate in rectangle[::2]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (top, bottom) for x0, top, x1, bottom in rectangles
            if x0 < right and left < x1
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for top, bottom in intervals[1:]:
                if top > end:
                    covered += end - start
                    start, end = top, bottom
                else:
                    end = max(end, bottom)
            covered += end - start
        area += (right - left) * covered
    return area


def _intersection_union_area(
    rectangle: tuple[float, float, float, float],
    others: Sequence[tuple[float, float, float, float]],
) -> float:
    intersections = []
    for other in others:
        overlap = (
            max(rectangle[0], other[0]), max(rectangle[1], other[1]),
            min(rectangle[2], other[2]), min(rectangle[3], other[3]),
        )
        if overlap[0] < overlap[2] and overlap[1] < overlap[3]:
            intersections.append(overlap)
    return _rect_union_area(intersections)


class SearchStateCollector:
    """Callable sink that keeps DTO traces separate from render adapters."""

    def __init__(self, original_image: Image.Image) -> None:
        if not isinstance(original_image, Image.Image):
            raise TypeError("original_image must be a PIL image")
        self._render_source = _RenderSource.capture(original_image)
        self._source_identity = _source_identity(original_image)
        self._source_image_key = _source_key(self._source_identity)
        self._candidates: dict[str, NextCandidate] = {}
        self._visited: set[str] = set()
        self._visited_renderer_identities: set[str] = set()
        self._snapshots: list[dict[str, Any]] = []
        self._rejected: list[dict[str, Any]] = []
        self._candidate_observations = 0

    def __call__(self, live_refs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        if not isinstance(live_refs, Mapping) or not isinstance(snapshot, Mapping):
            raise TypeError("search state callback requires mappings")
        try:
            frozen_snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("search state snapshot must be strict JSON") from error

        if frozen_snapshot.get("schema_version") != 1:
            raise ValueError("search state schema_version must be 1")
        event = frozen_snapshot.get("event")
        if event not in {"tree_ready", "stage_ready", "stage_finished", "p0_selected"}:
            raise ValueError("search state event is invalid")
        if frozen_snapshot.get("source_image_identity") != self._source_identity:
            raise ValueError("search state source image identity does not match collector image")

        candidates = frozen_snapshot.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("search state candidates must be a list")
        if frozen_snapshot.get("candidate_count") != len(candidates):
            raise ValueError("search state candidate_count does not match candidates")

        refs_by_key: dict[str, Any] = {}
        for role_refs in live_refs.values():
            if not isinstance(role_refs, tuple):
                raise ValueError("search state live refs must be immutable tuples")
            for ref in role_refs:
                key = getattr(ref, "canonical_key", None)
                if isinstance(key, str):
                    refs_by_key.setdefault(key, ref)

        pending_candidates: list[NextCandidate] = []
        pending_rejections: list[dict[str, Any]] = []
        next_ordinal = len(self._candidates)
        for raw_candidate in candidates:
            if not isinstance(raw_candidate, dict):
                raise ValueError("search state candidate must be an object")
            key = raw_candidate.get("canonical_key")
            if not isinstance(key, str) or not key:
                raise ValueError("search state candidate key must be nonempty")
            bbox = _bbox(raw_candidate.get("bbox_original"))
            depth = _integer(raw_candidate.get("depth"), "candidate depth")
            render_level = _integer(raw_candidate.get("render_level"), "candidate render_level")
            if key != _canonical_key(bbox, depth, render_level):
                raise ValueError("search state candidate canonical key does not match geometry")
            source = raw_candidate.get("source")
            if (source is not None and not isinstance(source, str)) or source not in _NATIVE_SOURCES:
                raise ValueError("search state candidate source is not a supported visual source")

            ref = refs_by_key.get(key)
            if ref is None:
                raise ValueError("search state candidate is missing its frozen descriptor")
            if (
                tuple(getattr(ref, "bbox_original", ())) != bbox
                or getattr(ref, "depth", None) != depth
                or getattr(ref, "render_level", None) != render_level
                or getattr(ref, "tree_scope", None) != frozen_snapshot.get("tree_scope")
                or tuple(getattr(ref, "crop_origin", ())) != tuple(frozen_snapshot.get("crop_origin", ()))
                or getattr(ref, "source_image_key", None) != self._source_image_key
            ):
                raise ValueError("search state frozen descriptor does not match snapshot")

            x, y, width, height = bbox
            if width <= 0 or height <= 0:
                pending_rejections.append({
                    "canonical_key": key,
                    "bbox_original": list(bbox),
                    "reason": "non_positive_bbox",
                })
                continue
            if (
                x < 0 or y < 0
                or x + width > self._render_source.size[0]
                or y + height > self._render_source.size[1]
            ):
                pending_rejections.append({
                    "canonical_key": key,
                    "bbox_original": list(bbox),
                    "reason": "bbox_outside_source_image",
                })
                continue
            if key in self._candidates or any(item.canonical_key == key for item in pending_candidates):
                continue

            posterior = raw_candidate.get("posterior_score")
            if posterior is not None:
                if isinstance(posterior, bool) or not isinstance(posterior, Real):
                    raise ValueError("candidate posterior_score must be finite or null")
                posterior = float(posterior)
                if not math.isfinite(posterior):
                    raise ValueError("candidate posterior_score must be finite or null")
            renderer_kind = _renderer_kind(source)
            pending_candidates.append(NextCandidate(
                canonical_key=key,
                bbox_original=bbox,
                depth=depth,
                render_level=render_level,
                posterior_score=posterior,
                first_seen_ordinal=next_ordinal,
                tree_scope=ref.tree_scope,
                crop_origin=tuple(ref.crop_origin),
                source_image_key=ref.source_image_key,
                source=source,
                renderer_kind=renderer_kind,
                renderer_identity=_renderer_identity(
                    ref.source_image_key, bbox, render_level, renderer_kind,
                ),
                _render_source=self._render_source,
            ))
            next_ordinal += 1

        visited = set(frozen_snapshot.get("popped_keys", ()))
        visited.update(frozen_snapshot.get("selected_keys", ()))
        if not all(isinstance(key, str) for key in visited):
            raise ValueError("search state visited keys must be strings")

        self._snapshots.append(frozen_snapshot)
        self._candidate_observations += len(candidates)
        self._rejected.extend(pending_rejections)
        self._candidates.update((item.canonical_key, item) for item in pending_candidates)
        self._visited.update(visited)
        self._visited_renderer_identities.update(
            self._candidates[key].renderer_identity
            for key in visited
            if key in self._candidates
        )

    def next_candidate(self) -> NextDecision:
        canonical_remaining = [
            candidate for key, candidate in self._candidates.items()
            if key not in self._visited
        ]
        remaining = [
            candidate for candidate in canonical_remaining
            if candidate.renderer_identity not in self._visited_renderer_identities
        ]
        if remaining:
            remaining.sort(key=lambda candidate: (
                candidate.posterior_score is None,
                0.0 if candidate.posterior_score is None else -candidate.posterior_score,
                candidate.first_seen_ordinal,
                candidate.canonical_key,
            ))
            selected = remaining[0]
            self._visited.add(selected.canonical_key)
            self._visited_renderer_identities.add(selected.renderer_identity)
            return NextDecision(candidate=replace(selected), no_op_reason=None)
        if canonical_remaining:
            reason = NextNoOpReason.ALL_OBSERVATIONS_VISITED
        elif self._candidates:
            reason = NextNoOpReason.ALL_CANDIDATES_VISITED
        elif self._candidate_observations:
            reason = NextNoOpReason.NO_VALID_CANDIDATES
        else:
            reason = NextNoOpReason.EMPTY_QUEUE
        return NextDecision(candidate=None, no_op_reason=reason)

    def peek_expand_candidate(self, current_keys: Sequence[str]) -> ExpandDecision:
        """Return one spatial context without mutating any collector state."""
        if isinstance(current_keys, (str, bytes)):
            return ExpandDecision(None, ExpandNoOpReason.UNBOUND_FOCUS, ())
        keys = tuple(current_keys)
        if not keys:
            return ExpandDecision(None, ExpandNoOpReason.EMPTY_FOCUS, keys)
        if (
            not all(isinstance(key, str) and key for key in keys)
            or any(key not in self._candidates for key in keys)
        ):
            return ExpandDecision(None, ExpandNoOpReason.UNBOUND_FOCUS, keys)
        if len(keys) != len(set(keys)):
            return ExpandDecision(None, ExpandNoOpReason.DUPLICATE_CONTEXT, keys)
        focus = tuple(self._candidates[key] for key in keys)
        if any(item.renderer_kind == "root" or item.source == "global" for item in focus):
            return ExpandDecision(None, ExpandNoOpReason.NONLOCAL_FOCUS, keys)
        focus_renderer_ids = {item.renderer_identity for item in focus}
        if len(focus_renderer_ids) != len(focus):
            return ExpandDecision(None, ExpandNoOpReason.DUPLICATE_CONTEXT, keys)

        focus_rectangles = tuple(_xyxy(item) for item in focus)
        focus_union = (
            min(item[0] for item in focus_rectangles),
            min(item[1] for item in focus_rectangles),
            max(item[2] for item in focus_rectangles),
            max(item[3] for item in focus_rectangles),
        )
        diagonal = math.hypot(*self._render_source.size)
        ranked: list[
            tuple[tuple[float, bool, float, int, str], NextCandidate, float]
        ] = []
        duplicate_seen = False
        no_new_area_seen = False
        for key, candidate in self._candidates.items():
            if key in keys:
                continue
            if candidate.renderer_identity in focus_renderer_ids:
                duplicate_seen = True
                continue
            if key in self._visited or candidate.renderer_identity in self._visited_renderer_identities:
                continue
            if candidate.renderer_kind == "root" or candidate.source == "global":
                continue
            rectangle = _xyxy(candidate)
            candidate_area = (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])
            outside_area = candidate_area - _intersection_union_area(
                rectangle, focus_rectangles,
            )
            if outside_area <= 0.0:
                no_new_area_seen = True
                continue
            if any(
                rectangle[0] >= item[0] and rectangle[1] >= item[1]
                and rectangle[2] <= item[2] and rectangle[3] <= item[3]
                for item in focus_rectangles
            ):
                no_new_area_seen = True
                continue
            if all(
                rectangle[0] <= item[0] and rectangle[1] <= item[1]
                and rectangle[2] >= item[2] and rectangle[3] >= item[3]
                for item in focus_rectangles
            ):
                continue
            edge_gap = min(
                math.hypot(
                    max(item[0] - rectangle[2], rectangle[0] - item[2], 0.0),
                    max(item[1] - rectangle[3], rectangle[1] - item[3], 0.0),
                )
                for item in focus_rectangles
            ) / diagonal
            rank = (
                edge_gap,
                candidate.posterior_score is None,
                0.0 if candidate.posterior_score is None else -candidate.posterior_score,
                candidate.first_seen_ordinal,
                candidate.canonical_key,
            )
            ranked.append((rank, candidate, outside_area))
        if not ranked:
            reason = (
                ExpandNoOpReason.ADDS_NO_NEW_AREA if no_new_area_seen
                else ExpandNoOpReason.DUPLICATE_CONTEXT if duplicate_seen
                else ExpandNoOpReason.NO_SPATIAL_CONTEXT
            )
            return ExpandDecision(
                None, reason, keys,
                focus_descriptors=tuple(replace(item) for item in focus),
            )
        rank, candidate, outside_area = min(ranked, key=lambda item: item[0])
        return ExpandDecision(
            candidate=replace(candidate), no_op_reason=None, current_keys=keys,
            focus_descriptors=tuple(replace(item) for item in focus),
            focus_union_xyxy=focus_union, positive_outside_area=outside_area,
            normalized_edge_gap=rank[0], rank_tuple=rank,
        )

    def support_view(self, node_keys: Sequence[str]) -> tuple[NextCandidate, ...] | None:
        if isinstance(node_keys, (str, bytes)):
            raise TypeError("node_keys must be a sequence of canonical keys")
        keys = tuple(node_keys)
        if not all(isinstance(key, str) for key in keys):
            return None
        if not keys:
            return ()
        if any(key not in self._candidates for key in keys):
            return None
        return tuple(replace(self._candidates[key]) for key in keys)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "source_image_identity": json.loads(json.dumps(self._source_identity)),
            "snapshots": json.loads(json.dumps(self._snapshots, allow_nan=False)),
            "candidates": [
                candidate.to_dict()
                for candidate in sorted(
                    self._candidates.values(), key=lambda item: item.first_seen_ordinal
                )
            ],
            "visited_keys": sorted(self._visited),
            "visited_renderer_identities": sorted(self._visited_renderer_identities),
            "rejected_candidates": json.loads(json.dumps(self._rejected, allow_nan=False)),
        }
        json.dumps(payload, allow_nan=False)
        return payload
