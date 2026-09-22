"""Answer-free grounding and stable target-instance identities for Revision 3."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

from PIL import Image

from qavs.evidence_gap.answers import canonical_text
from qavs.evidence_gap.pdf_types import SearchStateRecord
from qavs.evidence_gap.provenance import canonical_sha256


GROUNDING_LABELS = ("Grounded", "NotGrounded", "Insufficient")
_GROUNDING_PROMPT = (
    "Decide whether this image directly and spatially grounds the named target role "
    "from the question. Do not answer the question and do not use candidate answers. "
    "Classify with exactly one code: A = Grounded, B = NotGrounded, "
    "C = Insufficient. Return only A, B, or C.\n"
    "Question: {question}\nTarget role: {role}"
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not (result := " ".join(value.split())):
        raise ValueError(f"{name} must be nonempty text")
    return result


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return result


def _bbox(value: Any, name: str) -> tuple[float, float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 4
    ):
        raise ValueError(f"{name} must contain four numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result) or min(result[2:]) <= 0.0:
        raise ValueError(f"{name} must be a finite positive-area xywh box")
    return result  # type: ignore[return-value]


def _intersection(first: Sequence[float], second: Sequence[float]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    overlap = _intersection(first, second)
    first_area = first[2] * first[3]
    second_area = second[2] * second[3]
    return overlap / (first_area + second_area - overlap)


def _contains(outer: Sequence[float], inner: Sequence[float]) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih


@dataclass(frozen=True)
class GroundingLabelDistribution:
    labels: tuple[str, str, str]
    losses: tuple[float, float, float]
    probabilities: tuple[float, float, float]
    winner: str

    def __post_init__(self) -> None:
        if self.labels != GROUNDING_LABELS or self.winner not in self.labels:
            raise ValueError("grounding distribution has invalid frozen labels")
        if len(self.losses) != 3 or not all(math.isfinite(item) for item in self.losses):
            raise ValueError("grounding distribution requires three finite losses")
        if len(self.probabilities) != 3 or any(
            not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in self.probabilities
        ) or not math.isclose(
            math.fsum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("grounding probabilities must be normalized")

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "losses": list(self.losses),
            "probabilities": list(self.probabilities),
            "winner": self.winner,
        }


@dataclass(frozen=True)
class RoleGrounding:
    role: str
    role_sha256: str
    distribution: GroundingLabelDistribution
    grounded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "role_sha256": self.role_sha256,
            "distribution": self.distribution.to_dict(),
            "grounded": self.grounded,
        }


@dataclass(frozen=True)
class GroundingVerification:
    roles: tuple[RoleGrounding, ...]
    valid: bool
    model_calls: int
    processed_pixels: int
    checkpoint_sha256: str
    prompt_template_sha256: str

    def distributions(self) -> dict[str, GroundingLabelDistribution]:
        return {item.role: item.distribution for item in self.roles}

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": [item.to_dict() for item in self.roles],
            "valid": self.valid,
            "model_calls": self.model_calls,
            "processed_pixels": self.processed_pixels,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prompt_template_sha256": self.prompt_template_sha256,
        }


@dataclass(frozen=True)
class TargetInstance:
    instance_id: str
    role: str
    bbox_original: tuple[float, float, float, float]
    source: str
    proposal_key: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "role": self.role,
            "bbox_original": list(self.bbox_original),
            "source": self.source,
            "proposal_key": self.proposal_key,
        }


@dataclass(frozen=True)
class GroundingRecord:
    role_distributions: tuple[tuple[str, GroundingLabelDistribution], ...]
    role_to_instance: tuple[tuple[str, str], ...]
    covered_instance_ids: tuple[str, ...]
    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_distributions": [
                {"role": role, "distribution": distribution.to_dict()}
                for role, distribution in self.role_distributions
            ],
            "role_to_instance": [
                {"role": role, "instance_id": instance_id}
                for role, instance_id in self.role_to_instance
            ],
            "covered_instance_ids": list(self.covered_instance_ids),
            "valid": self.valid,
        }


def verify_target_grounding(
    *,
    question: str,
    roles: tuple[str, ...],
    image: Image.Image,
    conditional_losses: Callable[
        [Image.Image, str, tuple[str, str, str]],
        tuple[int, Sequence[float], int],
    ],
    checkpoint_sha256: str,
    generator_checkpoint_sha256: str,
    grounding_threshold: float,
) -> GroundingVerification:
    question = _text(question, "question")
    if (
        not isinstance(roles, tuple)
        or not roles
        or any(not isinstance(role, str) or not role.strip() for role in roles)
    ):
        raise ValueError("grounding roles must be a nonempty immutable text tuple")
    roles = tuple(_text(role, "grounding role") for role in roles)
    if len({canonical_text(role) for role in roles}) != len(roles):
        raise ValueError("grounding roles must be unique after normalization")
    if not isinstance(image, Image.Image):
        raise TypeError("grounding image must be a PIL image")
    if not callable(conditional_losses):
        raise TypeError("conditional_losses must be callable")
    checkpoint = _digest(checkpoint_sha256, "verifier checkpoint")
    generator = _digest(generator_checkpoint_sha256, "generator checkpoint")
    if checkpoint == generator:
        raise ValueError("grounding verifier must use a different checkpoint")
    threshold = _probability(grounding_threshold, "grounding_threshold")

    results = []
    model_calls = 0
    for role in roles:
        prompt = _GROUNDING_PROMPT.format(question=question, role=role)
        raw = conditional_losses(image, prompt, GROUNDING_LABELS)
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            raise ValueError("grounding verifier must return winner, losses, and calls")
        winner, losses, calls = raw
        if (
            isinstance(losses, (str, bytes))
            or not isinstance(losses, Sequence)
            or len(losses) != 3
        ):
            raise ValueError("grounding verifier requires three finite losses")
        values = tuple(float(item) for item in losses)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("grounding verifier requires three finite losses")
        expected = min(range(3), key=values.__getitem__)
        if type(winner) is not int or winner != expected:
            raise ValueError("grounding verifier winner disagrees with losses")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
            raise ValueError("grounding verifier calls must be positive")
        minimum = min(values)
        weights = tuple(math.exp(-(item - minimum)) for item in values)
        total = math.fsum(weights)
        probabilities = tuple(item / total for item in weights)
        distribution = GroundingLabelDistribution(
            labels=GROUNDING_LABELS,
            losses=values,
            probabilities=probabilities,
            winner=GROUNDING_LABELS[winner],
        )
        results.append(RoleGrounding(
            role=role,
            role_sha256=hashlib.sha256(role.encode("utf-8")).hexdigest(),
            distribution=distribution,
            grounded=(winner == 0 and probabilities[0] >= threshold),
        ))
        model_calls += calls
    return GroundingVerification(
        roles=tuple(results),
        valid=all(item.grounded for item in results),
        model_calls=model_calls,
        processed_pixels=model_calls * image.width * image.height,
        checkpoint_sha256=checkpoint,
        prompt_template_sha256=hashlib.sha256(
            _GROUNDING_PROMPT.encode("utf-8")
        ).hexdigest(),
    )


class TargetInstanceRegistry:
    def __init__(
        self,
        *,
        image: Image.Image,
        targets: tuple[str, ...],
        candidates: Sequence[Mapping[str, Any]],
        sam_model: Any,
        target_instance_iou: float,
    ) -> None:
        self._image = image.copy()
        self._targets = tuple(_text(item, "target role") for item in targets)
        self._sam_model = sam_model
        self._target_instance_iou = _probability(
            target_instance_iou, "target_instance_iou"
        )
        self._candidates: dict[str, dict[str, Any]] = {}
        self._instances: dict[str, TargetInstance] = {}
        self._proposal_aliases: dict[str, set[str]] = {}
        self._lazy_cache: dict[str, tuple[str, ...]] = {}
        self._grounded_views: dict[tuple[str, ...], tuple[str, ...]] = {}
        self.add_candidates(candidates)

    @classmethod
    def from_frontend(
        cls,
        image: Image.Image,
        targets: tuple[str, ...],
        candidates: Sequence[Mapping[str, Any]],
        sam_model: Any,
        target_instance_iou: float,
    ) -> "TargetInstanceRegistry":
        if not isinstance(image, Image.Image):
            raise TypeError("registry image must be a PIL image")
        return cls(
            image=image,
            targets=targets,
            candidates=candidates,
            sam_model=sam_model,
            target_instance_iou=target_instance_iou,
        )

    def _add_instance(
        self,
        *,
        role: str,
        bbox_original: Sequence[float],
        source: str,
        proposal_key: str | None,
    ) -> TargetInstance:
        role = _text(role, "instance role")
        bbox = _bbox(bbox_original, "instance bbox")
        matches = [
            item for item in self._instances.values()
            if canonical_text(item.role) == canonical_text(role)
            and _iou(item.bbox_original, bbox) >= self._target_instance_iou
        ]
        if matches:
            instance = min(matches, key=lambda item: item.instance_id)
            if proposal_key is not None:
                self._proposal_aliases[instance.instance_id].add(proposal_key)
            return instance
        instance_id = canonical_sha256({
            "role": canonical_text(role),
            "bbox_original": list(bbox),
            "source": source,
            "proposal_key": proposal_key,
        })
        instance = TargetInstance(
            instance_id=instance_id,
            role=role,
            bbox_original=bbox,
            source=source,
            proposal_key=proposal_key,
        )
        self._instances[instance_id] = instance
        self._proposal_aliases[instance_id] = (
            set() if proposal_key is None else {proposal_key}
        )
        return instance

    def add_candidates(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise TypeError("registry candidates must be a sequence")
        for raw in candidates:
            if not isinstance(raw, Mapping):
                raise TypeError("registry candidate must be a mapping")
            key = _text(raw.get("canonical_key"), "candidate key")
            bbox = _bbox(raw.get("bbox_original"), "candidate bbox")
            candidate = {
                "canonical_key": key,
                "bbox_original": bbox,
                "source": raw.get("source"),
                "sam_target_id": raw.get("sam_target_id"),
                "sam_target": raw.get("sam_target"),
                "sam_has_mask": raw.get("sam_has_mask"),
            }
            existing = self._candidates.get(key)
            if existing is not None and existing != candidate:
                raise ValueError("candidate metadata changed for a stable key")
            self._candidates[key] = candidate
            if candidate["source"] == "sam_proposal" and candidate["sam_target"]:
                self._add_instance(
                    role=candidate["sam_target"],
                    bbox_original=bbox,
                    source="global_sam",
                    proposal_key=key,
                )

    def instance(self, instance_id: str) -> TargetInstance:
        try:
            return self._instances[instance_id]
        except KeyError as error:
            raise KeyError(f"unknown target instance: {instance_id}") from error

    @staticmethod
    def _view_key(state: SearchStateRecord) -> tuple[str, ...]:
        return tuple(dict.fromkeys(state.focus_keys + state.context_keys))

    def target_bbox_for_state(
        self, state: SearchStateRecord,
    ) -> tuple[float, float, float, float] | None:
        identifiers = self._grounded_views.get(self._view_key(state))
        if not identifiers:
            return None
        boxes = [self._instances[item].bbox_original for item in identifiers]
        left = min(item[0] for item in boxes)
        top = min(item[1] for item in boxes)
        right = max(item[0] + item[2] for item in boxes)
        bottom = max(item[1] + item[3] for item in boxes)
        return left, top, right - left, bottom - top

    def _view_bbox(
        self,
        state: SearchStateRecord,
        effective_geometry: Sequence[float] | None = None,
    ) -> tuple[float, float, float, float]:
        if effective_geometry is not None:
            return _bbox(effective_geometry, "effective grounding geometry")
        keys = tuple(dict.fromkeys(state.focus_keys + state.context_keys))
        boxes = [self._candidates[key]["bbox_original"] for key in keys if key in self._candidates]
        if not boxes:
            raise ValueError("grounded state has no catalog geometry")
        left = min(item[0] for item in boxes)
        top = min(item[1] for item in boxes)
        right = max(item[0] + item[2] for item in boxes)
        bottom = max(item[1] + item[3] for item in boxes)
        return left, top, right - left, bottom - top

    def _matching_instance(
        self, role: str, state: SearchStateRecord, view_bbox: Sequence[float],
    ) -> TargetInstance | None:
        path = set(state.path_keys + state.focus_keys + state.context_keys)
        matches = [
            item for item in self._instances.values()
            if canonical_text(item.role) == canonical_text(role)
            and (
                bool(self._proposal_aliases[item.instance_id] & path)
                or _contains(view_bbox, item.bbox_original)
                or _contains(item.bbox_original, view_bbox)
                or _iou(view_bbox, item.bbox_original) >= self._target_instance_iou
            )
        ]
        if not matches:
            return None
        return min(matches, key=lambda item: (
            not bool(self._proposal_aliases[item.instance_id] & path),
            item.bbox_original[2] * item.bbox_original[3],
            item.instance_id,
        ))

    def _native_lazy_sam(
        self, crop: Image.Image, roles: tuple[str, ...],
    ) -> Mapping[str, Sequence[Sequence[float]]]:
        if self._sam_model is None:
            return {}
        result = self._sam_model.batch_inference(crop, list(roles))
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("lazy SAM must return three values")
        _, processed, target_ids = result
        if not isinstance(processed, Mapping) or not isinstance(target_ids, Sequence):
            raise ValueError("lazy SAM result has invalid mappings")
        output = {}
        for index, target_id in enumerate(target_ids):
            role = roles[index] if index < len(roles) else None
            row = processed.get(target_id)
            if role is None or not isinstance(row, Mapping):
                continue
            boxes = row.get("boxes", ())
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().tolist()
            elif hasattr(boxes, "tolist"):
                boxes = boxes.tolist()
            output[role] = boxes
        return output

    def _materialize_lazy(
        self,
        state: SearchStateRecord,
        roles: tuple[str, ...],
        view_bbox: tuple[float, float, float, float],
        lazy_sam: Callable[
            [Image.Image, tuple[str, ...]], Mapping[str, Sequence[Sequence[float]]]
        ] | None,
    ) -> None:
        cache_key = canonical_sha256({
            "pixels": hashlib.sha256(self._image.tobytes()).hexdigest(),
            "view_bbox": list(view_bbox),
            "roles": [canonical_text(role) for role in roles],
        })
        if cache_key in self._lazy_cache:
            return
        left, top, width, height = view_bbox
        crop = self._image.crop((
            round(left), round(top), round(left + width), round(top + height),
        ))
        callback = lazy_sam or self._native_lazy_sam
        raw = callback(crop, roles)
        if not isinstance(raw, Mapping):
            raise TypeError("lazy SAM callback must return a role-to-boxes mapping")
        added = []
        for role in roles:
            boxes = raw.get(role, ())
            if isinstance(boxes, (str, bytes)) or not isinstance(boxes, Sequence):
                raise TypeError("lazy SAM boxes must be a sequence")
            for raw_box in boxes:
                if not isinstance(raw_box, Sequence) or len(raw_box) < 4:
                    continue
                x0, y0, x1, y1 = map(float, raw_box[:4])
                x0, y0 = max(0.0, x0), max(0.0, y0)
                x1, y1 = min(float(crop.width), x1), min(float(crop.height), y1)
                if not all(math.isfinite(item) for item in (x0, y0, x1, y1)):
                    continue
                if x1 <= x0 or y1 <= y0:
                    continue
                instance = self._add_instance(
                    role=role,
                    bbox_original=(left + x0, top + y0, x1 - x0, y1 - y0),
                    source="lazy_local_sam",
                    proposal_key=None,
                )
                added.append(instance.instance_id)
        self._lazy_cache[cache_key] = tuple(sorted(set(added)))

    def ground_state(
        self,
        state: SearchStateRecord,
        role_distributions: Mapping[str, GroundingLabelDistribution],
        *,
        grounding_threshold: float,
        effective_geometry: Sequence[float] | None = None,
        lazy_sam: Callable[
            [Image.Image, tuple[str, ...]], Mapping[str, Sequence[Sequence[float]]]
        ] | None = None,
    ) -> GroundingRecord:
        if not isinstance(state, SearchStateRecord):
            raise TypeError("ground_state requires a SearchStateRecord")
        if not isinstance(role_distributions, Mapping) or not role_distributions:
            raise ValueError("role_distributions must be a nonempty mapping")
        threshold = _probability(grounding_threshold, "grounding_threshold")
        distributions = tuple(role_distributions.items())
        if any(
            not isinstance(role, str)
            or not isinstance(distribution, GroundingLabelDistribution)
            for role, distribution in distributions
        ):
            raise TypeError("role distributions contain an invalid entry")
        view_bbox = self._view_bbox(state, effective_geometry)
        passed = tuple(
            role for role, distribution in distributions
            if (
                distribution.winner == "Grounded"
                and distribution.probabilities[0] >= threshold
            )
        )
        mapping: dict[str, str] = {}
        missing = []
        for role in passed:
            instance = self._matching_instance(role, state, view_bbox)
            if instance is None:
                missing.append(role)
            else:
                mapping[role] = instance.instance_id
        if missing:
            self._materialize_lazy(
                state, tuple(missing), view_bbox, lazy_sam,
            )
            for role in missing:
                instance = self._matching_instance(role, state, view_bbox)
                if instance is not None:
                    mapping[role] = instance.instance_id
        covered = tuple(sorted(
            item.instance_id for item in self._instances.values()
            if any(canonical_text(item.role) == canonical_text(role) for role in passed)
            and (
                _contains(view_bbox, item.bbox_original)
                or _intersection(view_bbox, item.bbox_original) > 0.0
            )
        ))
        record = GroundingRecord(
            role_distributions=distributions,
            role_to_instance=tuple(
                (role, mapping[role]) for role, _ in distributions if role in mapping
            ),
            covered_instance_ids=covered,
            valid=(bool(mapping) and len(passed) == len(mapping)),
        )
        if record.valid:
            self._grounded_views[self._view_key(state)] = tuple(
                instance_id for _, instance_id in record.role_to_instance
            )
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": list(self._targets),
            "target_instance_iou": self._target_instance_iou,
            "instances": [{
                **item.to_dict(),
                "proposal_keys": sorted(self._proposal_aliases[item.instance_id]),
            } for item in sorted(
                self._instances.values(), key=lambda value: value.instance_id,
            )],
            "lazy_cache_keys": sorted(self._lazy_cache),
        }


def compatible_confirmation(
    first: GroundingRecord,
    second: GroundingRecord,
    *,
    question_kind: str,
    required_roles: tuple[str, ...] = (),
) -> bool:
    if not isinstance(first, GroundingRecord) or not isinstance(second, GroundingRecord):
        raise TypeError("confirmation compatibility requires grounding records")
    if question_kind not in {
        "attribute", "relation", "comparison", "count", "coverage",
    }:
        raise ValueError("question_kind is unsupported")
    if not first.valid or not second.valid:
        return False
    if question_kind == "attribute":
        return first.role_to_instance == second.role_to_instance
    required = {canonical_text(role) for role in required_roles}
    grounded = {
        canonical_text(role)
        for record in (first, second) for role, _ in record.role_to_instance
    }
    if required and not required.issubset(grounded):
        return False
    if question_kind in {"count", "coverage"}:
        instances = {
            instance_id
            for record in (first, second)
            for role, instance_id in record.role_to_instance
            if not required or canonical_text(role) in required
        }
        return len(instances) >= 2
    return True
