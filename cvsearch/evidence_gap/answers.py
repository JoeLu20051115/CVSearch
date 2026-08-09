"""Label-blind answer stability aggregation for benchmark-facing outputs."""

import math
import re
from collections.abc import Sequence
from fractions import Fraction
from numbers import Real

from .types import AnswerRecord


_OPTION_LINE = re.compile(r"^\s*([A-D])\.\s*(.*?)\s*$")
_EVALUATOR_LETTERS = frozenset("ABCD")


def _canonical_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def parse_option_block(block: str) -> dict[str, str]:
    """Parse a strict HR option block, retaining its evaluator letter mapping.

    Blank lines are ignored. Every nonblank line must be an uppercase ``A.`` to
    ``D.`` option with nonempty text, and labels may not repeat.
    """
    if not isinstance(block, str):
        raise TypeError("option block must be a string")
    options: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        match = _OPTION_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed option line: {line!r}")
        label, text = match.groups()
        text = " ".join(text.split())
        if not text:
            raise ValueError(f"empty option text for {label}")
        if label in options:
            raise ValueError(f"duplicate option label: {label}")
        options[label] = text
    if not options:
        raise ValueError("option block is empty")
    return options


def _official_letter(raw_output: str) -> str | None:
    """Match HR-Bench's evaluator: a one-character value or first A-D char."""
    if len(raw_output) == 1:
        return raw_output if raw_output in _EVALUATOR_LETTERS else None
    return next((char for char in raw_output if char in _EVALUATOR_LETTERS), None)


def aggregate_hr_answers(option_blocks: list[str], raw_outputs: list[str]) -> AnswerRecord:
    """Vote across shuffled HR options without consulting annotation truth.

    Invalid model letters (including a valid evaluator letter absent from that
    block) are ignored. Ties choose the earliest valid vote, then canonical
    semantic text for a stable fallback. Only the winning semantic answer must
    have exactly one reverse label in every block. A non-projectable winner (or
    no valid votes) fails closed to the exact raw evaluator outputs.
    """
    if len(option_blocks) != len(raw_outputs):
        raise ValueError("option_blocks and raw_outputs must have equal length")
    if not option_blocks:
        raise ValueError("at least one option block is required")
    if any(not isinstance(raw_output, str) for raw_output in raw_outputs):
        raise TypeError("raw output must be a string")

    semantic_maps: list[dict[str, str]] = []
    reverse_maps: list[dict[str, list[str]]] = []
    for block in option_blocks:
        options = parse_option_block(block)
        semantic_map = {label: _canonical_text(text) for label, text in options.items()}
        semantic_maps.append(semantic_map)
        reverse_map: dict[str, list[str]] = {}
        for label, semantic in semantic_map.items():
            reverse_map.setdefault(semantic, []).append(label)
        reverse_maps.append(reverse_map)

    groups: dict[str, dict[str, object]] = {}
    for index, (raw_output, semantic_map) in enumerate(zip(raw_outputs, semantic_maps)):
        letter = _official_letter(raw_output)
        if letter not in semantic_map:
            continue
        semantic = semantic_map[letter]
        group = groups.setdefault(semantic, {"count": 0, "first_vote": index, "vote_indices": []})
        group["count"] = int(group["count"]) + 1
        group["vote_indices"].append(index)

    if not groups:
        return AnswerRecord(
            output=list(raw_outputs),
            raw_outputs=tuple(raw_outputs),
            groups={},
            frequency=0.0,
            margin=0.0,
            confidence=0.0,
            uncertainty=1.0,
            aggregation_available=False,
            aggregation_reason="no_valid_votes",
        )

    winner = min(
        groups,
        key=lambda semantic: (-int(groups[semantic]["count"]), int(groups[semantic]["first_vote"]), semantic),
    )
    winning_count = int(groups[winner]["count"])
    runner_count = max((int(group["count"]) for semantic, group in groups.items() if semantic != winner), default=0)
    frequency = winning_count / len(option_blocks)
    margin = (winning_count - runner_count) / len(option_blocks)
    winner_labels = [reverse_map.get(winner, []) for reverse_map in reverse_maps]
    if any(len(labels) != 1 for labels in winner_labels):
        return AnswerRecord(
            output=list(raw_outputs),
            canonical_answer=None,
            raw_outputs=tuple(raw_outputs),
            groups=groups,
            frequency=frequency,
            margin=margin,
            confidence=0.0,
            uncertainty=1.0,
            aggregation_available=False,
            aggregation_reason="ambiguous_winner_projection",
        )
    return AnswerRecord(
        output=[labels[0] for labels in winner_labels],
        canonical_answer=winner,
        raw_outputs=tuple(raw_outputs),
        groups=groups,
        frequency=frequency,
        margin=margin,
        confidence=frequency,
        uncertainty=1.0 - frequency,
        aggregation_available=True,
        aggregation_reason=None,
    )


def _finite_loss(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("losses must be finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("losses must be finite numbers")
    return number


def _finite_mean(values: Sequence[float]) -> float:
    """Compute a finite mean while preserving cancellation and avoiding overflow."""
    try:
        mean = math.fsum(values) / len(values)
    except ArithmeticError as error:
        try:
            mean = float(sum((Fraction.from_float(value) for value in values), Fraction()) / len(values))
        except (ArithmeticError, ValueError) as fallback_error:
            raise ValueError("mean losses must be finite") from fallback_error
    if not math.isfinite(mean):
        raise ValueError("mean losses must be finite")
    return mean


def aggregate_vstar_losses(loss_rows: list[list[float]]) -> AnswerRecord:
    """Aggregate per-prompt option losses into V*'s integer option schema.

    The confidence margin is ``(second - best) / (abs(second) + abs(best))``;
    a zero denominator gives zero. This remains finite for zero or negative
    losses. A single option is certain by construction, and equal means choose
    the lower option index.
    """
    if not loss_rows:
        raise ValueError("loss_rows must be nonempty")
    if not all(isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and row for row in loss_rows):
        raise ValueError("each loss row must be nonempty")
    option_count = len(loss_rows[0])
    if any(len(row) != option_count for row in loss_rows):
        raise ValueError("loss rows must have equal option counts")
    rows = tuple(tuple(_finite_loss(value) for value in row) for row in loss_rows)
    means = tuple(_finite_mean(tuple(row[index] for row in rows)) for index in range(option_count))
    winner = min(range(option_count), key=lambda index: means[index])
    if option_count == 1:
        margin = 1.0
    else:
        runner = min(index for index in range(option_count) if index != winner)
        for index in range(option_count):
            if index != winner and means[index] < means[runner]:
                runner = index
        scale = max(abs(means[runner]), abs(means[winner]))
        if scale == 0.0:
            margin = 0.0
        else:
            scaled_runner = means[runner] / scale
            scaled_winner = means[winner] / scale
            margin = (scaled_runner - scaled_winner) / (abs(scaled_runner) + abs(scaled_winner))
        if not math.isfinite(margin):
            raise ValueError("loss margin must be finite")
        margin = min(1.0, max(0.0, margin))
    return AnswerRecord(
        output=winner,
        canonical_answer=winner,
        raw_outputs=rows,
        groups={"prompt_count": len(rows), "option_means": list(means)},
        frequency=1.0,
        margin=margin,
        confidence=margin,
        uncertainty=1.0 - margin,
        losses=means,
    )
