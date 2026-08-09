"""Pure HR soft-evidence fusion anchored to raw CVSearch outputs."""

import math
from collections.abc import Sequence
from numbers import Real

from .answers import aggregate_hr_answers, official_letter, parse_option_block
from .types import AnswerRecord


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def soft_fuse_hr(
    option_blocks: Sequence[str], raw_outputs: Sequence[str], evidence: AnswerRecord, gamma: float
) -> AnswerRecord:
    """Fuse global semantic votes into HR letters without overriding raw ties."""
    weight = _nonnegative_finite(gamma, "gamma")
    blocks = tuple(option_blocks)
    raw = tuple(raw_outputs)
    if len(blocks) != len(raw) or not blocks:
        raise ValueError("option blocks and raw outputs must be paired and nonempty")
    semantic_maps = tuple(parse_option_block(block) for block in blocks)
    if evidence.aggregation_available is not True or not evidence.groups:
        return AnswerRecord(
            output=list(raw),
            raw_outputs=raw,
            selected_from="cvsearch_anchor",
            aggregation_available=False,
            aggregation_reason=evidence.aggregation_reason,
        )
    valid_votes = sum(int(group["count"]) for group in evidence.groups.values())
    if valid_votes <= 0:
        raise ValueError("semantic evidence must contain a positive vote count")
    vote_probability = {
        semantic: int(group["count"]) / len(blocks)
        for semantic, group in evidence.groups.items()
    }
    fused = []
    for raw_output, semantic_map in zip(raw, semantic_maps):
        raw_letter = official_letter(raw_output)
        if raw_letter not in semantic_map:
            return AnswerRecord(
                output=list(raw),
                raw_outputs=raw,
                selected_from="cvsearch_anchor",
                aggregation_available=False,
                aggregation_reason="invalid_raw_anchor",
            )
        scores = {
            letter: float(letter == raw_letter) + weight * vote_probability.get(semantic, 0.0)
            for letter, semantic in semantic_map.items()
        }
        best_score = max(scores.values())
        winners = [letter for letter, score in scores.items() if score == best_score]
        selected = raw_letter if raw_letter in winners else min(winners)
        fused.append(raw_output if selected == raw_letter else selected)
    record = aggregate_hr_answers(list(blocks), fused)
    record.output = list(fused)
    record.raw_outputs = raw
    record.selected_from = "cvsearch_anchor" if fused == list(raw) else "unified_soft_fusion"
    return record
