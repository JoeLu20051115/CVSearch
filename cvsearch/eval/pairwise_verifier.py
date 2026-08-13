#!/usr/bin/env python3
"""Label-blind proposal and order-symmetric pairwise verification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from .replay_adaptive_search import FrozenCalibration, FrozenSelectedCalibration
from .replay_split_search import _split_audit
from .replay_uncertainty_support import (
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
    _canonical_json,
    _prepare_replay,
    candidate_snapshots,
    sanitize_replay_row,
)


PROPOSAL_OBSERVATIONS = 8
PROPOSAL_AGREEMENTS = (0.4, 0.5, 0.6)
CONFIDENCE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
_COMPARISON_CHOICES = (
    "Answer 1 is better supported.",
    "Answer 2 is better supported.",
)
_SHEET_SIZE = 896
_PANEL_SIZE = 448
_SHEET_BACKGROUND = (127, 127, 127)
_VIEW_COLORS = ((235, 64, 52), (49, 116, 229))


@dataclass(frozen=True)
class PairwiseProposal:
    feasible: bool
    stage2_output: Any
    candidate_output: Any
    p0_canonical: Any
    candidate_canonical: Any
    observations: int
    agreement: float
    agreeing_hashes: tuple[str, ...]


@dataclass(frozen=True)
class PairwiseProjection:
    feasible: bool
    confidence: float
    candidate_probabilities: tuple[float, float]

    def __post_init__(self) -> None:
        if type(self.feasible) is not bool:
            raise TypeError("pairwise feasibility must be an exact boolean")
        values = (self.confidence, *self.candidate_probabilities)
        if len(self.candidate_probabilities) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in values
        ):
            raise ValueError("pairwise probabilities must be finite in [0, 1]")


@dataclass(frozen=True)
class PairwiseEvidenceView:
    branch_index: int
    role: str
    crop_xyxy: tuple[int, int, int, int]
    source_size: tuple[int, int]
    render_sha256: str
    raw_support: float

    def __post_init__(self) -> None:
        if type(self.branch_index) is not int or self.branch_index < 0:
            raise ValueError("evidence branch index must be nonnegative")
        if self.role not in {"tight", "medium", "context"}:
            raise ValueError("evidence role is invalid")
        if (
            len(self.crop_xyxy) != 4
            or any(type(value) is not int for value in self.crop_xyxy)
            or len(self.source_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.source_size)
        ):
            raise ValueError("evidence geometry is invalid")
        x0, y0, x1, y1 = self.crop_xyxy
        width, height = self.source_size
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("evidence crop exceeds source size")
        if (
            not isinstance(self.render_sha256, str)
            or len(self.render_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.render_sha256)
        ):
            raise ValueError("evidence render hash is invalid")
        if (
            isinstance(self.raw_support, bool)
            or not isinstance(self.raw_support, (int, float))
            or not math.isfinite(float(self.raw_support))
            or not 0.0 <= float(self.raw_support) <= 1.0
        ):
            raise ValueError("evidence raw support must be finite in [0, 1]")
        object.__setattr__(self, "raw_support", float(self.raw_support))


def _dummy_policy() -> UnifiedPolicy:
    return UnifiedPolicy(
        profile="balanced",
        threshold=0.0,
        raw_support_floor=0.0,
        utility_calibrator=UtilityIsotonicCalibrator((1.0,), (0.5,)),
    )


def _stop_observations(prepared: Any) -> int:
    return sum(
        next(
            (
                index
                for index, view in enumerate(branch.views, start=1)
                if view.canonical_answer is None
            ),
            len(branch.views),
        )
        for branch in prepared.branches
    )


def propose_pairwise_candidate(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    calibration: FrozenCalibration | FrozenSelectedCalibration,
    *,
    minimum_agreement: float,
) -> PairwiseProposal:
    """Propose one latest-checkpoint candidate without evaluator metadata."""
    if minimum_agreement not in PROPOSAL_AGREEMENTS:
        raise ValueError("proposal agreement is outside the frozen grid")
    stage2 = sanitize_replay_row(stage2_row)
    split = sanitize_replay_row(split_row)
    prepared = _prepare_replay(stage2, split, calibration)
    observations = min(PROPOSAL_OBSERVATIONS, _stop_observations(prepared))
    eligible = [
        snapshot
        for snapshot in candidate_snapshots(
            stage2, split, calibration, _dummy_policy(),
        )
        if snapshot.observations <= PROPOSAL_OBSERVATIONS
        and len(snapshot.agreeing_hashes) >= 2
        and snapshot.features.agreement >= minimum_agreement
    ]
    if eligible:
        latest = max(snapshot.observations for snapshot in eligible)
        eligible = [
            snapshot for snapshot in eligible
            if snapshot.observations == latest
        ]
    if not eligible:
        return PairwiseProposal(
            False, copy.deepcopy(prepared.stage2["selected_output"]),
            None, copy.deepcopy(prepared.p0_canonical), None,
            observations, 0.0, (),
        )
    selected = min(eligible, key=lambda snapshot: (
        -snapshot.features.agreement,
        -snapshot.evidence_features.mean_support,
        -snapshot.features.conflict_margin_01,
        -len(snapshot.agreeing_hashes),
        _canonical_json(snapshot.canonical_answer),
    ))
    return PairwiseProposal(
        True,
        copy.deepcopy(prepared.stage2["selected_output"]),
        copy.deepcopy(selected.output),
        copy.deepcopy(prepared.p0_canonical),
        copy.deepcopy(selected.canonical_answer),
        selected.observations,
        selected.features.agreement,
        selected.agreeing_hashes,
    )


def select_pairwise_evidence_views(
    split_row: Mapping[str, Any], proposal: PairwiseProposal,
) -> tuple[PairwiseEvidenceView, PairwiseEvidenceView]:
    """Bind the two strongest agreeing render hashes to frozen crop geometry."""
    if not isinstance(proposal, PairwiseProposal) or not proposal.feasible:
        raise ValueError("pairwise evidence requires a feasible proposal")
    split = sanitize_replay_row(split_row)
    audit = _split_audit(split)
    if not isinstance(audit, Mapping):
        raise ValueError("pairwise evidence lacks a SPLIT audit")
    branches = audit.get("branches")
    if not isinstance(branches, list):
        raise ValueError("pairwise evidence branches are invalid")
    wanted = set(proposal.agreeing_hashes)
    found = []
    for branch_index, branch in enumerate(branches):
        if not isinstance(branch, Mapping):
            raise ValueError("pairwise evidence branch is invalid")
        for role in ("tight", "medium", "context"):
            value = branch.get(f"{role}_view")
            if not isinstance(value, Mapping):
                continue
            digest = value.get("render_sha256")
            if digest not in wanted:
                continue
            found.append(PairwiseEvidenceView(
                branch_index=branch_index,
                role=role,
                crop_xyxy=tuple(value.get("crop_xyxy", ())),
                source_size=tuple(value.get("source_size", ())),
                render_sha256=digest,
                raw_support=value.get("raw_support"),
            ))
    if len(found) != len(wanted) or len(found) < 2:
        raise ValueError("agreeing render hashes do not bind exactly to views")
    found.sort(key=lambda view: (
        -view.raw_support, view.branch_index, view.role, view.render_sha256,
    ))
    return found[0], found[1]


def _image_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _letterbox(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", size, _SHEET_BACKGROUND)
    panel.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return panel


def compose_pairwise_evidence_sheet(
    source: Image.Image,
    views: Sequence[PairwiseEvidenceView],
) -> tuple[Image.Image, dict[str, Any]]:
    """Render one marked overview above two answer-agreeing crops."""
    if type(source) is not Image.Image or source.mode != "RGB":
        raise TypeError("pairwise evidence source must be an exact RGB image")
    if (
        not isinstance(views, (list, tuple))
        or len(views) != 2
        or not all(type(view) is PairwiseEvidenceView for view in views)
    ):
        raise ValueError("pairwise evidence requires exactly two frozen views")
    frozen = tuple(views)
    if any(view.source_size != source.size for view in frozen):
        raise ValueError("pairwise evidence source size drifted")
    if len({view.render_sha256 for view in frozen}) != 2:
        raise ValueError("pairwise evidence views must be distinct")

    marked = source.copy()
    draw = ImageDraw.Draw(marked)
    thickness = max(2, min(source.size) // 160)
    for view, color in zip(frozen, _VIEW_COLORS):
        x0, y0, x1, y1 = view.crop_xyxy
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=thickness)
    panels = [_letterbox(marked, (_SHEET_SIZE, _PANEL_SIZE))]
    for view, color in zip(frozen, _VIEW_COLORS):
        crop = source.crop(view.crop_xyxy)
        panel = _letterbox(crop, (_PANEL_SIZE, _PANEL_SIZE))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle(
            (0, 0, _PANEL_SIZE - 1, _PANEL_SIZE - 1),
            outline=color, width=6,
        )
        panels.append(panel)
    sheet = Image.new("RGB", (_SHEET_SIZE, _SHEET_SIZE), _SHEET_BACKGROUND)
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (0, _PANEL_SIZE))
    sheet.paste(panels[2], (_PANEL_SIZE, _PANEL_SIZE))
    return sheet, {
        "view_sha256": [view.render_sha256 for view in frozen],
        "crop_xyxy": [list(view.crop_xyxy) for view in frozen],
        "source_size": list(source.size),
        "panel_sha256": [_image_sha256(panel) for panel in panels],
        "sheet_sha256": _image_sha256(sheet),
        "sheet_size": list(sheet.size),
    }


def compose_independent_source_view(
    source: Image.Image,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return a byte-identical copy of the complete source image."""
    if type(source) is not Image.Image or source.mode != "RGB":
        raise TypeError("independent source view must be an exact RGB image")
    view = source.copy()
    return view, {
        "view_sha256": _image_sha256(view),
        "view_size": list(view.size),
        "view_mode": view.mode,
    }


def _display(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def pairwise_answer_display(options: Any, answer: Any) -> str:
    """Resolve an inference answer into visible option text when possible."""
    if (
        type(answer) is int
        and isinstance(options, list)
        and 0 <= answer < len(options)
        and isinstance(options[answer], str)
        and options[answer].strip()
    ):
        return " ".join(options[answer].split())
    if isinstance(answer, str) and answer.strip():
        normalized = " ".join(answer.split())
        if isinstance(options, str) and len(normalized) == 1:
            prefix = f"{normalized.upper()}."
            for line in options.splitlines():
                line = " ".join(line.split())
                if line.upper().startswith(prefix):
                    return line
        return normalized
    return _display(answer)


def pairwise_prompt_material(
    question: str, options: Any, p0: Any, candidate: Any,
) -> dict[str, Any]:
    """Build two answer-order reversals over the same inference inputs."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("pairwise question must be nonempty")
    option_text = _display(options)

    def prompt(first: Any, second: Any) -> str:
        return (
            "Compare two proposed answers using only visible evidence in the "
            "image. Do not assume either proposal is correct.\n"
            f"Question: {question.strip()}\n"
            f"Options: {option_text}\n"
            f"Answer 1: {_display(first)}\n"
            f"Answer 2: {_display(second)}\n"
            "Which proposed answer is better supported?"
        )

    prompts = (prompt(p0, candidate), prompt(candidate, p0))
    return {
        "prompts": list(prompts),
        "choices": list(_COMPARISON_CHOICES),
        "candidate_choice_indices": [1, 0],
        "prompt_sha256": [
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in prompts
        ],
    }


def source_pairwise_prompt_material(
    question: str, p0: Any, candidate: Any,
) -> dict[str, Any]:
    """Build short-choice reversals for independent complete-image evidence."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("source pairwise question must be nonempty")

    def prompt(first: Any, second: Any) -> str:
        return (
            "Use only directly visible evidence in this complete image. Compare "
            "the two proposed answers without assuming either is correct.\n"
            f"Question: {question.strip()}\n"
            f"Proposal 1: {first}\n"
            f"Proposal 2: {second}\n"
            "Reply with exactly 1 if Proposal 1 is better supported, or exactly "
            "2 if Proposal 2 is better supported. Do not explain."
        )

    prompts = (
        prompt(pairwise_answer_display([], p0), pairwise_answer_display([], candidate)),
        prompt(pairwise_answer_display([], candidate), pairwise_answer_display([], p0)),
    )
    return {
        "prompts": list(prompts),
        "choices": ["1", "2"],
        "candidate_choice_indices": [1, 0],
        "prompt_sha256": [
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in prompts
        ],
    }


def _observation(value: Any, index: int) -> tuple[int, tuple[float, float]]:
    if type(value) is not dict or set(value) != {"winner", "losses"}:
        raise ValueError(f"pairwise observation {index} has an invalid schema")
    winner = value["winner"]
    losses = value["losses"]
    if type(winner) is not int or winner not in (0, 1):
        raise ValueError(f"pairwise observation {index} winner is invalid")
    if type(losses) is not list or len(losses) != 2:
        raise ValueError(f"pairwise observation {index} losses are invalid")
    normalized = []
    for loss in losses:
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
        ):
            raise ValueError("pairwise losses must be finite")
        normalized.append(float(loss))
    if winner != min(range(2), key=normalized.__getitem__):
        raise ValueError("pairwise winner differs from loss argmin")
    return winner, tuple(normalized)


def project_pairwise_losses(
    observations: Sequence[Mapping[str, Any]],
) -> PairwiseProjection:
    """Map reversed comparisons onto one normalized candidate confidence."""
    if isinstance(observations, (str, bytes)) or len(observations) != 2:
        raise ValueError("pairwise verification requires two observations")
    candidate_indices = (1, 0)
    probabilities = []
    winners = []
    for index, raw in enumerate(observations):
        winner, losses = _observation(raw, index)
        candidate = candidate_indices[index]
        other = 1 - candidate
        difference = losses[candidate] - losses[other]
        if difference >= 0.0:
            probability = math.exp(-difference) / (1.0 + math.exp(-difference))
        else:
            probability = 1.0 / (1.0 + math.exp(difference))
        probabilities.append(probability)
        winners.append(winner == candidate)
    confidence = min(probabilities)
    return PairwiseProjection(
        feasible=all(winners),
        confidence=confidence,
        candidate_probabilities=tuple(probabilities),
    )


def pairwise_decision(
    proposal: PairwiseProposal,
    projection: PairwiseProjection | None,
    *,
    agreement_threshold: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Return candidate only after the shared order-symmetric verifier gate."""
    if not isinstance(proposal, PairwiseProposal):
        raise TypeError("pairwise proposal must be frozen")
    if agreement_threshold not in PROPOSAL_AGREEMENTS:
        raise ValueError("pairwise agreement threshold is outside the grid")
    if confidence_threshold not in CONFIDENCE_THRESHOLDS:
        raise ValueError("pairwise confidence threshold is outside the grid")
    observed = proposal.observations + (2 if proposal.feasible else 0)
    selected = (
        proposal.feasible
        and proposal.agreement >= agreement_threshold
        and isinstance(projection, PairwiseProjection)
        and projection.feasible
        and projection.confidence >= confidence_threshold
    )
    return {
        "selected_output": copy.deepcopy(
            proposal.candidate_output if selected else proposal.stage2_output
        ),
        "selected_source": "PAIRWISE" if selected else "P0",
        "reason": (
            "order_symmetric_verification_reached"
            if selected else "pairwise_verification_rejected"
        ),
        "observations": observed,
    }


__all__ = [
    "CONFIDENCE_THRESHOLDS",
    "PROPOSAL_AGREEMENTS",
    "PROPOSAL_OBSERVATIONS",
    "PairwiseProjection",
    "PairwiseProposal",
    "compose_independent_source_view",
    "compose_pairwise_evidence_sheet",
    "pairwise_answer_display",
    "pairwise_decision",
    "pairwise_prompt_material",
    "project_pairwise_losses",
    "propose_pairwise_candidate",
    "select_pairwise_evidence_views",
    "source_pairwise_prompt_material",
]
