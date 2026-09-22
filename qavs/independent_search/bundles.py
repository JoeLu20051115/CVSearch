"""Query-grounded multi-target evidence bundle construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from PIL import Image

from qavs.evidence_gap.answers import canonical_text


_MULTI_TARGET_KINDS = {"relation", "comparison", "count", "coverage"}


@dataclass(frozen=True)
class EvidenceBundlePlan:
    question_kind: str
    required_roles: tuple[str, ...]
    constituent_state_ids: tuple[int, ...]
    role_to_instance: tuple[tuple[str, str], ...]
    covered_instance_ids: tuple[str, ...]
    coverage_context: bool
    valid: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_kind": self.question_kind,
            "required_roles": list(self.required_roles),
            "constituent_state_ids": list(self.constituent_state_ids),
            "role_to_instance": [
                {"role": role, "instance_id": instance_id}
                for role, instance_id in self.role_to_instance
            ],
            "covered_instance_ids": list(self.covered_instance_ids),
            "coverage_context": self.coverage_context,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _Candidate:
    state_id: int
    path: tuple[str, ...]
    geometry: tuple[float, float, float, float]
    render_sha256: str
    role_to_instance: tuple[tuple[str, str], ...]
    covered_instance_ids: tuple[str, ...]


def _text_tuple(values: Any, name: str) -> tuple[str, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
        or any(not isinstance(item, str) or not item.strip() for item in values)
    ):
        raise ValueError(f"{name} must be a nonempty text sequence")
    normalized = tuple(" ".join(item.split()) for item in values)
    if len({canonical_text(item) for item in normalized}) != len(normalized):
        raise ValueError(f"{name} must be unique after normalization")
    return normalized


def _candidate(record: Any) -> _Candidate:
    if not isinstance(record, Mapping):
        raise TypeError("bundle record must be a mapping")
    state = record.get("state")
    grounding = record.get("grounding")
    grounding_record = (
        grounding.get("record") if isinstance(grounding, Mapping) else None
    )
    if not isinstance(state, Mapping) or not isinstance(grounding_record, Mapping):
        raise TypeError("bundle record requires state and grounding mappings")
    state_id = state.get("state_id")
    if isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0:
        raise ValueError("bundle state id must be a non-negative integer")
    path = state.get("path_keys")
    if not isinstance(path, list) or not path or any(
        not isinstance(item, str) or not item for item in path
    ):
        raise ValueError("bundle path must be a nonempty string list")
    geometry = state.get("effective_geometry")
    if not isinstance(geometry, list) or len(geometry) != 4:
        raise ValueError("bundle geometry must be an xywh list")
    box = tuple(float(item) for item in geometry)
    if not all(math.isfinite(item) for item in box) or min(box[2:]) <= 0.0:
        raise ValueError("bundle geometry must have finite positive area")
    render = state.get("render_sha256")
    if not isinstance(render, str) or len(render) != 64 or any(
        item not in "0123456789abcdef" for item in render
    ):
        raise ValueError("bundle render identity must be a SHA-256 digest")
    mappings = grounding_record.get("role_to_instance")
    if not isinstance(mappings, list) or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("role"), str)
        or not item.get("role")
        or not isinstance(item.get("instance_id"), str)
        or not item.get("instance_id")
        for item in mappings
    ):
        raise ValueError("bundle grounding mappings are invalid")
    covered = grounding_record.get("covered_instance_ids")
    if not isinstance(covered, list) or any(
        not isinstance(item, str) or not item for item in covered
    ):
        raise ValueError("bundle covered instances are invalid")
    return _Candidate(
        state_id=state_id,
        path=tuple(path),
        geometry=box,  # type: ignore[arg-type]
        render_sha256=render,
        role_to_instance=tuple(
            (" ".join(item["role"].split()), item["instance_id"])
            for item in mappings
        ),
        covered_instance_ids=tuple(sorted(set(covered))),
    )


def build_bundle_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    question_kind: str,
    required_roles: Sequence[str],
    coverage_context: bool,
) -> EvidenceBundlePlan:
    """Select only grounded role/instance records needed by a multi-target query."""
    if question_kind not in _MULTI_TARGET_KINDS:
        raise ValueError("evidence bundles are restricted to multi-target questions")
    roles = _text_tuple(required_roles, "required_roles")
    if type(coverage_context) is not bool:
        raise TypeError("coverage_context must be boolean")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("bundle records must be a sequence")
    candidates = sorted((_candidate(item) for item in records), key=lambda item: item.state_id)
    latest_by_render: dict[str, _Candidate] = {}
    for item in candidates:
        if item.role_to_instance:
            latest_by_render[item.render_sha256] = item
    deduplicated = sorted(
        latest_by_render.values(), key=lambda item: item.state_id,
    )

    required = {canonical_text(role): role for role in roles}
    selected: dict[int, _Candidate] = {}
    role_mapping: dict[str, str] = {}
    if question_kind in {"relation", "comparison"}:
        matches_by_role = {
            role: tuple(
                (candidate, instance_id)
                for candidate in reversed(deduplicated)
                for mapped_role, instance_id in candidate.role_to_instance
                if canonical_text(mapped_role) == canonical_text(role)
            )
            for role in roles
        }

        def assign(
            index: int, used_state_ids: frozenset[int],
        ) -> tuple[tuple[str, _Candidate, str], ...] | None:
            if index == len(roles):
                return ()
            role = roles[index]
            for candidate, instance_id in matches_by_role[role]:
                if candidate.state_id in used_state_ids:
                    continue
                tail = assign(index + 1, used_state_ids | {candidate.state_id})
                if tail is not None:
                    return ((role, candidate, instance_id),) + tail
            return None

        assignment = assign(0, frozenset())
        if assignment is None:
            valid = False
            reason = (
                "distinct_role_branches_incomplete"
                if all(matches_by_role.values()) else "required_roles_incomplete"
            )
        else:
            valid, reason = True, "complete"
            for role, candidate, instance_id in assignment:
                selected[candidate.state_id] = candidate
                role_mapping[role] = instance_id
    else:
        instance_records: dict[str, _Candidate] = {}
        for candidate in deduplicated:
            for mapped_role, instance_id in candidate.role_to_instance:
                if canonical_text(mapped_role) in required:
                    instance_records[instance_id] = candidate
        for instance_id, candidate in sorted(
            instance_records.items(), key=lambda item: item[1].state_id,
        ):
            selected[candidate.state_id] = candidate
        instances = tuple(sorted(instance_records))
        if len(instances) < 2:
            valid, reason = False, "distinct_instances_incomplete"
        elif not coverage_context:
            valid, reason = False, "coverage_context_missing"
        else:
            valid, reason = True, "complete"
        if instances:
            role_mapping[roles[0]] = instances[0]

    ordered = tuple(selected[state_id] for state_id in sorted(selected))
    # Several grounded instances in one crop do not provide distinct observations.
    if valid and len(ordered) < 2:
        valid, reason = False, "distinct_constituent_states_incomplete"
    covered_instances = tuple(sorted({
        instance_id
        for item in ordered
        for instance_id in (
            item.covered_instance_ids
            + tuple(value for _, value in item.role_to_instance)
        )
        if (
            question_kind in {"relation", "comparison"}
            or instance_id in {
                value for candidate in ordered
                for role, value in candidate.role_to_instance
                if canonical_text(role) in required
            }
        )
    }))
    return EvidenceBundlePlan(
        question_kind=question_kind,
        required_roles=roles,
        constituent_state_ids=tuple(item.state_id for item in ordered),
        role_to_instance=tuple(
            (role, role_mapping[role]) for role in roles if role in role_mapping
        ),
        covered_instance_ids=covered_instances,
        coverage_context=coverage_context,
        valid=valid,
        reason=reason,
    )


def _fit(image: Image.Image, max_edge: int) -> Image.Image:
    edge = max(image.size)
    if edge <= max_edge:
        return image
    scale = max_edge / edge
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )


def render_evidence_bundle(
    image: Image.Image,
    records: Sequence[Mapping[str, Any]],
    plan: EvidenceBundlePlan,
) -> Image.Image:
    """Render a deterministic overview followed by the selected local crops."""
    if not isinstance(image, Image.Image):
        raise TypeError("bundle source must be a PIL image")
    if not isinstance(plan, EvidenceBundlePlan) or not plan.valid:
        raise ValueError("only a valid evidence bundle plan can be rendered")
    candidates = (_candidate(item) for item in records)
    by_state = {item.state_id: item for item in candidates}
    selected = [by_state[state_id] for state_id in plan.constituent_state_ids]
    overview = _fit(image.convert("RGB"), 1024)
    crops = []
    for item in selected:
        x, y, width, height = item.geometry
        left = max(0, math.floor(x))
        top = max(0, math.floor(y))
        right = min(image.width, math.ceil(x + width))
        bottom = min(image.height, math.ceil(y + height))
        if right <= left or bottom <= top:
            raise ValueError("bundle constituent crop is empty")
        crops.append(_fit(image.crop((left, top, right, bottom)).convert("RGB"), 1024))
    separator = 8
    crop_width = sum(item.width for item in crops) + separator * (len(crops) - 1)
    crop_height = max(item.height for item in crops)
    canvas = Image.new(
        "RGB",
        (max(overview.width, crop_width), overview.height + separator + crop_height),
        (122, 116, 104),
    )
    canvas.paste(overview, ((canvas.width - overview.width) // 2, 0))
    left = (canvas.width - crop_width) // 2
    for crop in crops:
        canvas.paste(crop, (left, overview.height + separator))
        left += crop.width + separator
    return _fit(canvas, 4096)
