"""Threshold-free cross-model answer duel on one joint evidence sheet."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from cvsearch.eval.phase7_uncertainty_confirmation import (
    _exact_rgb_image,
    _image_sha256,
    _p0,
    _snapshot_json,
)
from cvsearch.eval.phase9_dense_evidence_search import (
    DENSE_BOX_RGB,
    DENSE_PANEL_SIZE,
    DENSE_SEPARATOR_PIXELS,
    DenseTile,
    _candidate,
    _letterbox,
)


JOINT_PANEL_SIZE = DENSE_PANEL_SIZE
JOINT_SEPARATOR_PIXELS = DENSE_SEPARATOR_PIXELS
JOINT_BACKGROUND_RGB = (127, 127, 127)
JOINT_DUEL_PROMPT_VERSION = "joint_evidence_order_balanced_duel_v1"
JOINT_DUEL_TEMPLATE = (
    "Compare the two proposed answers using only directly visible evidence in the "
    "displayed image. Do not use prior knowledge and do not infer missing details.\n"
    "Question: {question}\n"
    "A. {answer_a}\n"
    "B. {answer_b}\n"
    "Which proposed answer is better supported by the visible evidence? Answer A or B."
)


@dataclass(frozen=True)
class JointDuelDecision:
    action: str
    status: str
    output: Any


def render_joint_evidence_sheet(
    source: Image.Image, tiles: Sequence[DenseTile],
) -> tuple[Image.Image, dict[str, Any]]:
    source = _exact_rgb_image(source, "joint evidence source")
    if (
        not isinstance(tiles, (list, tuple)) or len(tiles) != 3
        or not all(type(tile) is DenseTile for tile in tiles)
        or len({tile.identity for tile in tiles}) != 3
    ):
        raise ValueError("joint evidence requires three unique dense tiles")
    panels = []
    for tile in tiles:
        x0, y0, x1, y1 = tile.box
        if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
            raise ValueError("joint evidence tile exceeds source image")
        panels.append(_letterbox(source.crop(tile.box)))
    marked = source.copy()
    draw = ImageDraw.Draw(marked)
    thickness = max(2, min(source.size) // 200)
    for tile in tiles:
        x0, y0, x1, y1 = tile.box
        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1), outline=DENSE_BOX_RGB, width=thickness,
        )
    panels.append(_letterbox(marked))
    side = JOINT_PANEL_SIZE * 2 + JOINT_SEPARATOR_PIXELS
    sheet = Image.new("RGB", (side, side), JOINT_BACKGROUND_RGB)
    offset = JOINT_PANEL_SIZE + JOINT_SEPARATOR_PIXELS
    for panel, position in zip(panels, ((0, 0), (offset, 0), (0, offset), (offset, offset))):
        sheet.paste(panel, position)
    return sheet, {
        "tile_identities": [tile.identity for tile in tiles],
        "tile_boxes": [list(tile.box) for tile in tiles],
        "panel_sha256": [_image_sha256(panel) for panel in panels],
        "sheet_sha256": _image_sha256(sheet),
        "sheet_size": [side, side],
    }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not (value := " ".join(value.split())):
        raise ValueError(f"{name} must be nonempty text")
    return value


def duel_prompts(
    question: str, p0_answer: str, candidate_answer: str,
) -> dict[str, str]:
    question = _text(question, "question")
    p0_answer = _text(p0_answer, "P0 answer")
    candidate_answer = _text(candidate_answer, "candidate answer")
    return {
        "p0_first": JOINT_DUEL_TEMPLATE.format(
            question=question, answer_a=p0_answer, answer_b=candidate_answer,
        ),
        "candidate_first": JOINT_DUEL_TEMPLATE.format(
            question=question, answer_a=candidate_answer, answer_b=p0_answer,
        ),
    }


def _duel_observation(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"winner", "a_logit", "b_logit"}:
        raise ValueError(f"{name} has an invalid exact schema")
    winner = value["winner"]
    logits = (value["a_logit"], value["b_logit"])
    if type(winner) is not int or winner not in (0, 1):
        raise ValueError(f"{name} winner must be A or B")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        or not math.isfinite(float(item)) for item in logits
    ):
        raise ValueError(f"{name} logits must be finite")
    logits = tuple(float(item) for item in logits)
    if winner != min((0, 1), key=lambda index: (-logits[index], index)):
        raise ValueError(f"{name} winner disagrees with logits")
    return {"winner": winner, "a_logit": logits[0], "b_logit": logits[1]}


def _generator_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"winner", "losses"}:
        raise ValueError("generator duel has an invalid exact schema")
    winner = value["winner"]
    losses = value["losses"]
    if (
        type(winner) is not int or winner not in (0, 1)
        or not isinstance(losses, (list, tuple)) or len(losses) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) for item in losses
        )
    ):
        raise ValueError("generator duel winner and losses must be finite")
    losses = tuple(float(item) for item in losses)
    if winner != min((0, 1), key=lambda index: (losses[index], index)):
        raise ValueError("generator duel winner disagrees with losses")
    return {"winner": winner, "losses": list(losses)}


def project_joint_duel(
    *, p0_first: Any, candidate_first: Any, generator: Any,
) -> dict[str, Any]:
    p0_first = _duel_observation(p0_first, "P0-first duel")
    candidate_first = _duel_observation(candidate_first, "candidate-first duel")
    generator = _generator_observation(generator)
    feasible = (
        p0_first["winner"] == 1
        and candidate_first["winner"] == 0
        and generator["winner"] == 1
    )
    return {
        "feasible": feasible,
        "p0_first": p0_first,
        "candidate_first": candidate_first,
        "generator": generator,
    }


def select_joint_candidate(
    p0: dict[str, Any], candidate: dict[str, Any], projection: Mapping[str, Any],
) -> JointDuelDecision:
    p0, _ = _p0(p0)
    candidate, _ = _candidate(candidate)
    if not isinstance(projection, Mapping) or set(projection) != {
        "feasible", "p0_first", "candidate_first", "generator",
    }:
        raise ValueError("joint duel projection has an invalid exact schema")
    rebuilt = project_joint_duel(
        p0_first=projection["p0_first"],
        candidate_first=projection["candidate_first"],
        generator=projection["generator"],
    )
    if dict(projection) != rebuilt:
        raise ValueError("joint duel projection is not canonical")
    if (
        candidate["feasible"] and candidate["output"] != p0["output"]
        and rebuilt["feasible"]
    ):
        return JointDuelDecision(
            "DENSE", "selected_cross_model_unanimous", _snapshot_json(candidate["output"]),
        )
    return JointDuelDecision("P0", "retained_p0", _snapshot_json(p0["output"]))
