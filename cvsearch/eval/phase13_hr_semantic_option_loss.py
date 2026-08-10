"""Schema-normalized semantic option-loss projection for shuffled HR answers."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from cvsearch.evidence_gap.answers import (
    aggregate_vstar_losses,
    canonical_text,
    parse_option_block,
)
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.eval.phase7_uncertainty_confirmation import (
    _confidence,
    _exact_snapshot,
    _p0,
    _sha256,
    _snapshot_json,
)


HR_LOSS_MIN_PATH_CONSENSUS = frozenset({2 / 3, 1.0})
HR_LOSS_MIN_CONFIDENCES = frozenset({0.0, 0.05, 0.1, 0.25})
_CANDIDATE_FIELDS = frozenset({
    "action", "feasible", "output", "stability", "path_consensus",
    "view_sha256", "b5_record_sha256", "option_set_sha256",
})


@dataclass(frozen=True)
class HROptionLossDecision:
    action: str
    status: str
    output: Any
    confidence: float | None
    path_consensus: float | None


def build_semantic_option_set(option_blocks: Sequence[str]) -> dict[str, Any]:
    if (
        isinstance(option_blocks, (str, bytes))
        or not isinstance(option_blocks, Sequence) or len(option_blocks) != 4
    ):
        raise ValueError("HR semantic option reconstruction requires four blocks")
    parsed = [parse_option_block(block) for block in option_blocks]
    if any(tuple(block) != ("A", "B", "C", "D") for block in parsed):
        raise ValueError("each HR block must contain A through D exactly once")
    choices = [" ".join(value.split()) for value in parsed[0].values()]
    canonical = [canonical_text(value) for value in choices]
    if len(set(canonical)) != 4:
        raise ValueError("first HR block contains duplicate semantic choices")
    reverse = []
    expected = set(canonical)
    for block in parsed:
        mapping = {canonical_text(text): letter for letter, text in block.items()}
        if len(mapping) != 4 or set(mapping) != expected:
            raise ValueError("HR blocks must contain the same semantic choices")
        reverse.append(mapping)
    material = {
        "choices": choices,
        "canonical_choices": canonical,
        "letters_by_choice": [
            [mapping[value] for mapping in reverse] for value in canonical
        ],
        "option_blocks_sha256": canonical_sha256(list(option_blocks)),
    }
    material["identity_sha256"] = canonical_sha256(material)
    return material


def _loss_observation(value: Any, count: int, name: str):
    if not isinstance(value, Mapping) or set(value) != {"winner", "losses"}:
        raise ValueError(f"{name} has an invalid exact schema")
    winner, losses = value["winner"], value["losses"]
    if (
        type(winner) is not int or not 0 <= winner < count
        or type(losses) is not list or len(losses) != count
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(float(item)) for item in losses
        )
    ):
        raise ValueError(f"{name} has invalid winner/loss dimensions")
    losses = [float(item) for item in losses]
    if winner != min(range(count), key=losses.__getitem__):
        raise ValueError(f"{name} winner disagrees with losses")
    record = aggregate_vstar_losses([losses])
    if record.output != winner:
        raise ValueError(f"{name} aggregate winner changed")
    return record


def project_hr_loss_candidate(
    option_set: Mapping[str, Any], observations: Any,
) -> dict[str, Any]:
    if not isinstance(option_set, Mapping) or set(option_set) != {
        "choices", "canonical_choices", "letters_by_choice",
        "option_blocks_sha256", "identity_sha256",
    }:
        raise ValueError("HR semantic option set has an invalid exact schema")
    choices = option_set["choices"]
    letters = option_set["letters_by_choice"]
    if (
        type(choices) is not list or len(choices) != 4
        or type(letters) is not list or len(letters) != 4
    ):
        raise ValueError("HR semantic option set has invalid dimensions")
    if type(observations) is not list or len(observations) not in (2, 3):
        raise ValueError("HR option-loss projection requires two or three views")
    records = [
        _loss_observation(value, len(choices), f"HR option-loss view {index}")
        for index, value in enumerate(observations)
    ]
    outputs = [int(record.output) for record in records]
    winner, count = min(
        Counter(outputs).items(), key=lambda item: (-item[1], item[0]),
    )
    feasible = count >= 2
    majority = [record for record in records if record.output == winner]
    path_consensus = count / len(records) if feasible else 0.0
    confidence = min(
        [path_consensus] + [float(record.confidence) for record in majority],
    ) if feasible else 0.0
    return {
        "feasible": feasible,
        "output": _snapshot_json(letters[winner]) if feasible else None,
        "confidence": confidence,
        "semantic_winner": winner,
        "canonical_answer": option_set["canonical_choices"][winner],
        "path_consensus": path_consensus,
        "view_winners": outputs,
    }


def hr_loss_rule_key(min_path_consensus: float, min_confidence: float) -> str:
    if (
        type(min_path_consensus) not in (int, float)
        or float(min_path_consensus) not in HR_LOSS_MIN_PATH_CONSENSUS
        or type(min_confidence) not in (int, float)
        or float(min_confidence) not in HR_LOSS_MIN_CONFIDENCES
    ):
        raise ValueError("HR option-loss rule must use the frozen grid")
    return f"p{float(min_path_consensus)}-c{float(min_confidence)}"


def all_hr_loss_rule_keys() -> tuple[str, ...]:
    return tuple(
        hr_loss_rule_key(path, confidence)
        for path in sorted(HR_LOSS_MIN_PATH_CONSENSUS)
        for confidence in sorted(HR_LOSS_MIN_CONFIDENCES)
    )


def _candidate(value: Any) -> tuple[dict[str, Any], float | None]:
    candidate = _exact_snapshot(value, _CANDIDATE_FIELDS, "HR option-loss candidate")
    if candidate["action"] != "HR_LOSS" or type(candidate["feasible"]) is not bool:
        raise ValueError("HR option-loss candidate action or feasibility is invalid")
    _sha256(candidate["b5_record_sha256"], "B5 record hash")
    _sha256(candidate["option_set_sha256"], "HR option-set hash")
    hashes = candidate["view_sha256"]
    if type(hashes) is not list or len(hashes) > 3:
        raise ValueError("HR option-loss candidate view hashes are invalid")
    for index, value in enumerate(hashes):
        _sha256(value, f"HR option-loss view hash {index}")
    if len(set(hashes)) != len(hashes):
        raise ValueError("HR option-loss view hashes must be distinct")
    if not candidate["feasible"]:
        if any(candidate[field] is not None for field in ("output", "stability", "path_consensus")):
            raise ValueError("infeasible HR option-loss candidate exposes a projection")
        return candidate, None
    if len(hashes) not in (2, 3) or candidate["output"] is None:
        raise ValueError("feasible HR option-loss candidate lacks views or output")
    if candidate["path_consensus"] not in HR_LOSS_MIN_PATH_CONSENSUS:
        raise ValueError("HR option-loss candidate has invalid path consensus")
    return candidate, _confidence(candidate["stability"], "HR option-loss stability")


def select_hr_loss_candidate(
    p0: dict[str, Any], candidate: dict[str, Any], *,
    min_path_consensus: float, min_confidence: float,
) -> HROptionLossDecision:
    hr_loss_rule_key(min_path_consensus, min_confidence)
    p0, _ = _p0(p0)
    candidate, confidence = _candidate(candidate)
    retained = HROptionLossDecision(
        "P0", "retained_p0", _snapshot_json(p0["output"]), None, None,
    )
    if not candidate["feasible"] or candidate["output"] == p0["output"]:
        return retained
    if confidence is None:
        raise AssertionError("feasible HR option-loss candidate lost confidence")
    if (
        float(candidate["path_consensus"]) < float(min_path_consensus)
        or confidence < float(min_confidence)
    ):
        return retained
    return HROptionLossDecision(
        "HR_LOSS", "selected_hr_semantic_option_loss",
        _snapshot_json(candidate["output"]), confidence,
        float(candidate["path_consensus"]),
    )
