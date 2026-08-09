"""Deterministic, answer-free collection of CVSearch NEXT candidates."""

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

from cvsearch.models.tree import NodeA, NodeState


class NextNoOpReason(str, Enum):
    EMPTY_QUEUE = "next_queue_empty"
    NO_VALID_CANDIDATES = "next_no_valid_candidates"
    ALL_CANDIDATES_VISITED = "next_all_candidates_visited"


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
    _render_source: _RenderSource = field(repr=False, compare=False)

    @property
    def render_node(self) -> NodeA:
        """Build a fresh original-image adapter without copying tree state."""
        return NodeA(NodeState(
            original_image_pil=self._render_source.render_image(),
            bbox=list(self.bbox_original),
        ))

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
        }


@dataclass(frozen=True)
class NextDecision:
    candidate: NextCandidate | None
    no_op_reason: NextNoOpReason | None

    def __post_init__(self) -> None:
        if (self.candidate is None) == (self.no_op_reason is None):
            raise ValueError("NEXT decision must contain exactly one candidate or no-op reason")


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
        if event not in {"stage_ready", "stage_finished", "p0_selected"}:
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

    def next_candidate(self) -> NextDecision:
        remaining = [
            candidate for key, candidate in self._candidates.items()
            if key not in self._visited
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
            return NextDecision(candidate=replace(selected), no_op_reason=None)
        if self._candidates:
            reason = NextNoOpReason.ALL_CANDIDATES_VISITED
        elif self._candidate_observations:
            reason = NextNoOpReason.NO_VALID_CANDIDATES
        else:
            reason = NextNoOpReason.EMPTY_QUEUE
        return NextDecision(candidate=None, no_op_reason=reason)

    def support_view(self, node_keys: Sequence[str]) -> tuple[NextCandidate, ...] | None:
        if isinstance(node_keys, (str, bytes)):
            raise TypeError("node_keys must be a sequence of canonical keys")
        keys = tuple(node_keys)
        if not keys or not all(isinstance(key, str) for key in keys):
            return None
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
            "rejected_candidates": json.loads(json.dumps(self._rejected, allow_nan=False)),
        }
        json.dumps(payload, allow_nan=False)
        return payload
