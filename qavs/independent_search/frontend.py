"""Root-plus-SAM proposal front end with lazy SGAP recovery."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

import numpy as np
from PIL import Image

from qavs.evidence_gap.search_state import SearchStateCollector, _rect_union_area
from qavs.independent_search.config import ProposalConfig
from qavs.models.modeling_sam3 import ConstrainedTreeBuilder
from qavs.models.tree import AdaptiveImageTree
from qavs.models.utils import (
    extract_visual_objects,
    include_pronouns,
    normalize_target_text,
)


@dataclass(frozen=True)
class _FrozenCandidate:
    canonical_key: str
    bbox_original: tuple[int | float, int | float, int | float, int | float]
    depth: int
    render_level: int
    tree_scope: str
    crop_origin: tuple[int, int]
    source_image_key: str


@dataclass(frozen=True)
class ProposalCoverage:
    valid_count: int
    spatial_coverage: float
    min_count: int
    min_spatial_coverage: float
    adequate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_count": self.valid_count,
            "spatial_coverage": self.spatial_coverage,
            "min_count": self.min_count,
            "min_spatial_coverage": self.min_spatial_coverage,
            "adequate": self.adequate,
        }


@dataclass(frozen=True)
class RecoveryResult:
    trigger: str
    added_keys: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    collector_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "added_keys": list(self.added_keys),
            "duplicate_keys": list(self.duplicate_keys),
            "collector_sha256": self.collector_sha256,
        }


@dataclass(frozen=True)
class ProposalFrontendResult:
    image: Image.Image
    targets: tuple[str, ...]
    collector: SearchStateCollector
    proposal_keys: tuple[str, ...]
    coverage: ProposalCoverage
    recovery: RecoveryHandle
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "proposal_keys": list(self.proposal_keys),
            "coverage": self.coverage.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }

    def proposal_trace(self) -> dict[str, Any]:
        return {
            "targets": list(self.targets),
            "proposal_keys": list(self.proposal_keys),
            "coverage": self.coverage.to_dict(),
        }


# Keep collection-time compatibility until the schema-2 method replaces its import.
FrontendResult = ProposalFrontendResult


class _ProposalCollector(SearchStateCollector):
    """Adapt proposal provenance to the collector's existing fine renderer."""

    def __call__(self, live_refs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
        public_snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
        native_snapshot = json.loads(json.dumps(public_snapshot))
        for candidate in native_snapshot["candidates"]:
            if candidate.get("source") in {"sam_proposal", "sgap_recovered"}:
                candidate["source"] = "fine"
        super().__call__(live_refs, native_snapshot)
        # SearchStateCollector intentionally freezes JSON snapshots. Keep its
        # native render descriptors while restoring method-owned provenance.
        self._snapshots[-1] = public_snapshot


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _source_identity(image: Image.Image) -> dict[str, Any]:
    return {
        "mode": image.mode,
        "size": [image.width, image.height],
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def _number(value: Any, name: str = "candidate value") -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return int(number) if number.is_integer() else number


def _bbox(node: Any) -> list[int | float]:
    values = list(node.state.bbox)
    if len(values) != 4:
        raise ValueError("candidate bbox must have four values")
    return [_number(value, "tree bbox") for value in values]  # type: ignore[list-item]


def _key(bbox: Sequence[int | float], depth: int) -> str:
    normalized = [_number(value, "candidate bbox") for value in bbox]
    return _canonical_json({
        "bbox": normalized, "depth": int(depth), "render_level": 0,
    })


def _candidate(
    bbox: Sequence[int | float],
    *,
    depth: int,
    parent_key: str | None,
    child_keys: Sequence[str],
    source: str,
    rank: int,
    complexity: int | float | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    normalized_bbox = [_number(value, "candidate bbox") for value in bbox]
    result = {
        "canonical_key": _key(normalized_bbox, depth),
        "bbox_original": normalized_bbox,
        "parent_key": parent_key,
        "child_keys": list(child_keys),
        "depth": int(depth),
        "render_level": 0,
        "source": source,
        "stage_rank": rank,
        "prior_prob": None,
        "complexity": complexity,
        "fast_confidence": None,
        "posterior_score": None,
        "is_evaluated": False,
        "answering_confidence": None,
    }
    result.update(metadata)
    return result


def _refs(candidates: Sequence[Mapping[str, Any]], image: Image.Image) -> tuple[_FrozenCandidate, ...]:
    source_key = _canonical_json(_source_identity(image))
    return tuple(
        _FrozenCandidate(
            canonical_key=candidate["canonical_key"],
            bbox_original=tuple(candidate["bbox_original"]),
            depth=int(candidate["depth"]),
            render_level=int(candidate["render_level"]),
            tree_scope="main",
            crop_origin=(0, 0),
            source_image_key=source_key,
        )
        for candidate in candidates
    )


def _snapshot(
    candidates: Sequence[Mapping[str, Any]],
    image: Image.Image,
    targets: tuple[str, ...],
    ordinal: int,
) -> dict[str, Any]:
    copied = [dict(candidate) for candidate in candidates]
    keys = [candidate["canonical_key"] for candidate in copied]
    return {
        "schema_version": 1,
        "event": "tree_ready",
        "tree_scope": "main",
        "crop_origin": [0, 0],
        "source_image_identity": _source_identity(image),
        "search_call_ordinal": ordinal,
        "visual_cue": " and ".join(targets),
        "stage": "SAM Proposals" if ordinal == 1 else "SCAN/RECOVER",
        "depth": 0,
        "candidate_count": len(copied),
        "candidates": copied,
        "ordered_keys": keys,
        "popped_keys": [],
        "selected_keys": [],
        "remaining_keys": keys,
    }


def _emit(
    collector: SearchStateCollector,
    candidates: Sequence[Mapping[str, Any]],
    image: Image.Image,
    targets: tuple[str, ...],
    ordinal: int,
) -> None:
    collector(
        MappingProxyType({"ordered_nodes": _refs(candidates, image)}),
        _snapshot(candidates, image, targets, ordinal),
    )


def _build_tree(features: Any, image: Image.Image) -> AdaptiveImageTree:
    if hasattr(features, "detach"):
        features = features.detach().cpu().float().numpy()
    array = np.asarray(features)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("SAM vision_features must have shape CxHxW")
    builder = ConstrainedTreeBuilder(
        array, n_atoms=600, pos_weight=3.5, split_threshold=0.3,
        keep_threshold=0.15,
    )
    tree_dict = builder.build_tree(max_depth=3, min_splits=4, max_splits=8)
    return AdaptiveImageTree(image, tree_dict, array.shape)


_GENERIC_VISUAL_TARGET = re.compile(
    r"(?:(?:the|a|an|this|that)\s+)?(?:image|picture|photo)", re.IGNORECASE,
)


def presence_question_target(question: str) -> str | None:
    """Extract a plain presence target, including POPE's observed 'imange' typo."""
    match = re.fullmatch(
        r"(?:is there (?:a|an|any)|are there (?:any|some)) "
        r"([a-z][a-z'-]*(?: [a-z][a-z'-]*)*) "
        r"in (?:the|this) (?:image|imange|picture|photo)\?"
        r"(?: answer yes or no only\.?)?",
        " ".join(question.split()), re.IGNORECASE,
    )
    if match is None:
        return None
    target = match.group(1)
    words = target.lower().split()
    # ponytail: keep relations and quantifiers on the general question path.
    if set(words) & {
        "a", "an", "the", "this", "these", "those",
        "and", "or", "nor", "not", "no", "both", "either", "neither", "only",
        "all", "every", "each", "another", "other", "more", "less", "fewer",
        "than", "least", "most", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "of", "in", "on", "at", "to", "from",
        "for", "with", "without", "by", "near", "next", "beside", "behind",
        "under", "above", "below", "over", "between", "around", "through",
        "across", "inside", "outside", "that", "which", "who", "whose",
        "is", "are", "has", "have", "wearing", "holding", "carrying", "riding",
        "sitting", "standing", "lying", "playing", "looking", "facing",
        "located", "positioned", "attached",
    } or _GENERIC_VISUAL_TARGET.fullmatch(target):
        return None
    # A trailing participle can add a relation ("cat chasing mice"). Defer
    # ambiguous phrases; initial compounds like "dining table" stay supported.
    if any(len(word) > 3 and word.endswith(("ing", "ed")) for word in words[1:]):
        return None
    return target


def _targets(
    zoom_model: Any, nlp_model: Any, ic_examples: Any, question: str,
) -> tuple[str, ...]:
    presence_target = presence_question_target(question)
    if presence_target is not None:
        return (presence_target,)
    generated = zoom_model.generate_visual_cues_using_ic(ic_examples, question)
    if not isinstance(generated, (list, tuple)):
        raise TypeError("visual cues must be a sequence")
    usable = []
    for raw in generated:
        if not isinstance(raw, str) or not raw.strip() or include_pronouns(nlp_model, raw):
            continue
        normalized, _ = normalize_target_text(raw)
        normalized = " ".join(normalized.split())
        if normalized and not _GENERIC_VISUAL_TARGET.fullmatch(normalized):
            usable.append(normalized)
    if not usable:
        usable = [
            " ".join(value.split())
            for value in extract_visual_objects(nlp_model, question)
            if isinstance(value, str) and value.strip()
            and not _GENERIC_VISUAL_TARGET.fullmatch(normalize_target_text(value)[0])
        ]
    if not usable:
        usable = ["visible question evidence"]
    return tuple(dict.fromkeys(usable))


def _xyxy_to_xywh(
    box: Any, width: int, height: int,
) -> tuple[float, float, float, float] | None:
    try:
        left, top, right, bottom = map(float, box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    left, top = max(0.0, left), max(0.0, top)
    right, bottom = min(float(width), right), min(float(height), bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _iou(
    first: Sequence[int | float], second: Sequence[int | float],
) -> float:
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by),
    )
    return overlap / (aw * ah + bw * bh - overlap)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _freeze_features(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _json_id(value: Any) -> str | int | float | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    return str(value)


def _normalize_proposals(
    processed_results: Mapping[Any, Any],
    target_ids: Sequence[Any],
    targets: tuple[str, ...],
    image: Image.Image,
    dedup_iou: float,
) -> list[dict[str, Any]]:
    flattened: list[tuple[float | None, tuple[float, float, float, float], Any, int, bool]] = []
    for target_index, target_id in enumerate(target_ids):
        raw = processed_results.get(target_id)
        if not isinstance(raw, Mapping):
            continue
        boxes = _array(raw.get("boxes", ()))
        if boxes.size == 0:
            continue
        if boxes.ndim == 1 and boxes.size == 4:
            boxes = boxes.reshape(1, 4)
        if boxes.ndim != 2 or boxes.shape[1] < 4:
            continue
        scores = _array(raw.get("scores", ())).reshape(-1)
        masks = raw.get("masks")
        mask_count = 0 if masks is None else len(masks)
        for box_index, raw_box in enumerate(boxes):
            bbox = _xyxy_to_xywh(raw_box[:4], image.width, image.height)
            if bbox is None:
                continue
            score = None
            if box_index < scores.size:
                candidate_score = float(scores[box_index])
                if math.isfinite(candidate_score):
                    score = candidate_score
            flattened.append((
                score, bbox, target_id, target_index, box_index < mask_count,
            ))
    flattened.sort(key=lambda item: (
        item[0] is None,
        0.0 if item[0] is None else -item[0],
        item[1],
    ))
    kept: list[dict[str, Any]] = []
    for score, bbox, target_id, target_index, has_mask in flattened:
        if any(_iou(bbox, item["bbox"]) >= dedup_iou for item in kept):
            continue
        kept.append({
            "bbox": bbox,
            "score": score,
            "target_id": _json_id(target_id),
            "target": targets[target_index] if target_index < len(targets) else None,
            "has_mask": has_mask,
        })
    return kept


def _coverage(
    proposals: Sequence[Mapping[str, Any]],
    image: Image.Image,
    config: ProposalConfig,
) -> ProposalCoverage:
    rectangles = tuple(
        (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        for item in proposals
        for bbox in (tuple(map(float, item["bbox"])),)
    )
    fraction = _rect_union_area(rectangles) / float(image.width * image.height)
    return ProposalCoverage(
        valid_count=len(proposals),
        spatial_coverage=fraction,
        min_count=config.min_count,
        min_spatial_coverage=config.min_spatial_coverage,
        adequate=(
            len(proposals) >= config.min_count
            and fraction >= config.min_spatial_coverage
        ),
    )


class RecoveryHandle:
    def __init__(
        self,
        *,
        features: Any,
        image: Image.Image,
        targets: tuple[str, ...],
        initial_candidates: Sequence[Mapping[str, Any]],
        dedup_iou: float,
    ) -> None:
        self._features = _freeze_features(features)
        self._image = image.copy()
        self._targets = targets
        self._initial_candidates = tuple(
            json.loads(json.dumps(candidate, allow_nan=False))
            for candidate in initial_candidates
        )
        self._dedup_iou = dedup_iou
        self._result: RecoveryResult | None = None
        self._tree: Any = None
        self._build_attempted = False
        self._build_error: Exception | None = None
        self._pending: tuple[str, tuple[str, ...], tuple[str, ...], str] | None = None

    def _finalize_committed(
        self, collector: SearchStateCollector,
    ) -> RecoveryResult | None:
        if self._pending is None:
            return None
        trigger, added, duplicates, expected_snapshot = self._pending
        payload = collector.to_dict()
        if not any(
            _canonical_json(snapshot) == expected_snapshot
            for snapshot in payload["snapshots"]
        ):
            return None
        self._result = RecoveryResult(
            trigger,
            added,
            duplicates,
            hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
        )
        return self._result

    def materialize(
        self, collector: SearchStateCollector, *, trigger: str,
    ) -> RecoveryResult:
        if self._result is not None:
            return self._result
        if not isinstance(collector, SearchStateCollector):
            raise TypeError("collector must be SearchStateCollector")
        if not isinstance(trigger, str) or not trigger.strip():
            raise ValueError("recovery trigger must be nonempty")
        committed = self._finalize_committed(collector)
        if committed is not None:
            return committed

        candidates = [json.loads(json.dumps(item)) for item in self._initial_candidates]
        by_key = {item["canonical_key"]: item for item in candidates}
        known = [
            (tuple(map(float, item["bbox_original"])), item["canonical_key"])
            for item in candidates
        ]
        aliases: dict[int, str] = {}
        added: list[str] = []
        duplicates: list[str] = []
        if not self._build_attempted:
            self._build_attempted = True
            try:
                self._tree = _build_tree(self._features, self._image)
            except Exception as error:
                self._build_error = error
                raise
        if self._build_error is not None:
            raise RuntimeError("recovery tree construction previously failed") from self._build_error
        tree = self._tree
        queue = [tree.root]
        while queue:
            node = queue.pop(0)
            queue.extend(tuple(getattr(node, "children", ()) or ()))
            bbox = _bbox(node)
            node_key = _key(bbox, int(node.depth))
            match = next((key for prior, key in known if _iou(bbox, prior) >= self._dedup_iou), None)
            if match is not None:
                aliases[id(node)] = match
                duplicates.append(node_key)
                continue
            parent = getattr(node, "parent", None)
            parent_key = None if parent is None else aliases[id(parent)]
            candidate = _candidate(
                bbox,
                depth=int(node.depth),
                parent_key=parent_key,
                child_keys=(),
                source="sgap_recovered",
                rank=len(candidates),
                complexity=_number(getattr(node, "complexity", None), "tree complexity"),
            )
            candidates.append(candidate)
            by_key[node_key] = candidate
            known.append((tuple(map(float, bbox)), node_key))
            aliases[id(node)] = node_key
            added.append(node_key)

        queue = [tree.root]
        while queue:
            node = queue.pop(0)
            children = tuple(getattr(node, "children", ()) or ())
            queue.extend(children)
            parent_key = aliases[id(node)]
            parent_candidate = by_key[parent_key]
            parent_candidate["child_keys"] = list(dict.fromkeys(
                list(parent_candidate.get("child_keys") or ())
                + [aliases[id(child)] for child in children if aliases[id(child)] != parent_key]
            ))

        expected_snapshot = _canonical_json(
            _snapshot(candidates, self._image, self._targets, 2),
        )
        if self._pending is None:
            self._pending = (
                trigger.strip(), tuple(added), tuple(dict.fromkeys(duplicates)),
                expected_snapshot,
            )
        elif self._pending[3] != expected_snapshot:
            raise RuntimeError("recovery retry produced inconsistent candidates")
        _emit(collector, candidates, self._image, self._targets, 2)
        committed = self._finalize_committed(collector)
        if committed is None:
            raise RuntimeError("recovery snapshot was not committed")
        return committed


def materialize_frontend(
    *,
    image: Image.Image,
    question: str,
    ic_examples: Any,
    sam_model: Any,
    zoom_model: Any,
    nlp_model: Any,
    proposal_config: ProposalConfig,
) -> ProposalFrontendResult:
    """Materialize the full-image root and answer-free SAM proposal children."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    if not isinstance(proposal_config, ProposalConfig):
        raise TypeError("proposal_config must be ProposalConfig")

    targets = _targets(zoom_model, nlp_model, ic_examples, question)
    output = sam_model.batch_inference(image, list(targets))
    if not isinstance(output, tuple) or len(output) != 3:
        raise ValueError("SAM batch_inference must return three values")
    backbone_out, processed_results, target_ids = output
    if not isinstance(backbone_out, Mapping) or "vision_features" not in backbone_out:
        raise ValueError("SAM output lacks vision_features")
    if not isinstance(processed_results, Mapping):
        raise ValueError("SAM processed results must be a mapping")
    if isinstance(target_ids, (str, bytes)) or not isinstance(target_ids, Sequence):
        raise ValueError("SAM target ids must be a sequence")

    normalized = _normalize_proposals(
        processed_results, target_ids, targets, image, proposal_config.dedup_iou,
    )
    root_bbox = [0, 0, image.width, image.height]
    root_key = _key(root_bbox, 0)
    proposal_candidates = [
        _candidate(
            item["bbox"],
            depth=1,
            parent_key=root_key,
            child_keys=(),
            source="sam_proposal",
            rank=index + 1,
            complexity=0,
            sam_score=item["score"],
            sam_target_id=item["target_id"],
            sam_target=item["target"],
            sam_has_mask=item["has_mask"],
        )
        for index, item in enumerate(normalized)
    ]
    proposal_keys = tuple(item["canonical_key"] for item in proposal_candidates)
    candidates = [
        _candidate(
            root_bbox,
            depth=0,
            parent_key=None,
            child_keys=proposal_keys,
            source="global",
            rank=0,
            complexity=0,
        ),
        *proposal_candidates,
    ]
    collector = _ProposalCollector(image)
    _emit(collector, candidates, image, targets, 1)
    coverage = _coverage(normalized, image, proposal_config)
    recovery = RecoveryHandle(
        features=backbone_out["vision_features"],
        image=image,
        targets=targets,
        initial_candidates=candidates,
        dedup_iou=proposal_config.dedup_iou,
    )
    diagnostics = MappingProxyType({
        "mode": "root_plus_sam_proposals",
        "proposal_count": len(proposal_keys),
        "sam_target_count": len(target_ids),
    })
    return ProposalFrontendResult(
        image.copy(), targets, collector, proposal_keys, coverage, recovery,
        diagnostics,
    )
