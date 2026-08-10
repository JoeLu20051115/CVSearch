"""Independent paired visual-support selection for frozen dense candidates."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence

from cvsearch.eval.phase7_uncertainty_confirmation import _p0, _snapshot_json
from cvsearch.eval.phase9_dense_evidence_search import _candidate
from cvsearch.evidence_gap.answers import aggregate_hr_answers


PAIRED_SUPPORT_PROMPT_VERSION = "independent_paired_answer_support_v1"
PAIRED_SUPPORT_PROCESSOR_MODE = (
    "single_dense_sheet_chat_left_padding_final_yes_no_logits"
)
PAIRED_SUPPORT_TEMPLATE = (
    "Assess only whether the displayed visual observation directly supports the "
    "proposed answer to the question. Use visible evidence only. Do not use prior "
    "knowledge, do not infer missing details, and do not replace the proposed answer.\n"
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Does this visual observation directly support the proposed answer? Answer Yes or No."
)
PAIRED_AVG_MARGINS = frozenset({0.0, 0.05, 0.1})
PAIRED_MIN_MARGINS = frozenset({0.0, 0.05})
PAIRED_SHEETS = 3


@dataclass(frozen=True)
class PairedSupportStats:
    p0_avg: float
    p0_min: float
    candidate_avg: float
    candidate_min: float
    avg_delta: float
    min_delta: float
    candidate_sheet_wins: int


@dataclass(frozen=True)
class PairedVerifierDecision:
    action: str
    status: str
    output: Any
    avg_delta: float | None
    min_delta: float | None
    candidate_sheet_wins: int | None


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not (value := " ".join(value.split())):
        raise ValueError(f"{name} must be nonempty text")
    return value


def proposed_answer_text(
    answer_type: str, options: Sequence[str], output: Any,
) -> str | None:
    if answer_type == "logits_match":
        if (
            type(output) is not int or isinstance(options, (str, bytes))
            or not isinstance(options, Sequence) or not 0 <= output < len(options)
        ):
            raise ValueError("V* proposed answer is outside exact options")
        return _text(options[output], "V* proposed answer")
    if answer_type == "option_list":
        if type(options) is not list or type(output) is not list:
            raise TypeError("HR proposed answer requires exact option and output lists")
        record = aggregate_hr_answers(options, output)
        if record.aggregation_available is not True or record.canonical_answer is None:
            return None
        return _text(record.canonical_answer, "HR proposed answer")
    raise ValueError("paired verifier answer type is unsupported")


def verifier_prompt(question: str, proposed_answer: str) -> str:
    return PAIRED_SUPPORT_TEMPLATE.format(
        question=_text(question, "question"),
        answer=_text(proposed_answer, "proposed answer"),
    )


def _support(values: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != PAIRED_SHEETS:
        raise ValueError(f"{name} must contain exactly three probabilities")
    result = []
    for value in values:
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must contain finite probabilities")
        result.append(float(value))
    return tuple(result)  # type: ignore[return-value]


def paired_support_stats(p0_support: Any, candidate_support: Any) -> PairedSupportStats:
    p0_values = _support(p0_support, "P0 support")
    candidate_values = _support(candidate_support, "candidate support")
    p0_avg = statistics.fmean(p0_values)
    candidate_avg = statistics.fmean(candidate_values)
    return PairedSupportStats(
        p0_avg=p0_avg,
        p0_min=min(p0_values),
        candidate_avg=candidate_avg,
        candidate_min=min(candidate_values),
        avg_delta=candidate_avg - p0_avg,
        min_delta=min(candidate_values) - min(p0_values),
        candidate_sheet_wins=sum(
            candidate > current
            for current, candidate in zip(p0_values, candidate_values)
        ),
    )


def paired_rule_key(min_avg_delta: float, min_min_delta: float) -> str:
    if (
        type(min_avg_delta) not in (int, float)
        or float(min_avg_delta) not in PAIRED_AVG_MARGINS
        or type(min_min_delta) not in (int, float)
        or float(min_min_delta) not in PAIRED_MIN_MARGINS
    ):
        raise ValueError("paired verifier rule must use the frozen grid")
    return f"a{float(min_avg_delta)}-m{float(min_min_delta)}"


def _retained(p0: dict[str, Any], stats: PairedSupportStats | None = None):
    return PairedVerifierDecision(
        "P0", "retained_p0", _snapshot_json(p0["output"]),
        None if stats is None else stats.avg_delta,
        None if stats is None else stats.min_delta,
        None if stats is None else stats.candidate_sheet_wins,
    )


def select_paired_candidate(
    p0: dict[str, Any], candidate: dict[str, Any], *,
    p0_support: Any, candidate_support: Any,
    min_avg_delta: float, min_min_delta: float,
) -> PairedVerifierDecision:
    paired_rule_key(min_avg_delta, min_min_delta)
    p0, _ = _p0(p0)
    candidate, _ = _candidate(candidate)
    if not candidate["feasible"] or candidate["output"] == p0["output"]:
        return _retained(p0)
    stats = paired_support_stats(p0_support, candidate_support)
    if (
        stats.candidate_sheet_wins < 2
        or Decimal(str(stats.avg_delta)) <= Decimal(str(float(min_avg_delta)))
        or Decimal(str(stats.min_delta)) < Decimal(str(float(min_min_delta)))
    ):
        return _retained(p0, stats)
    return PairedVerifierDecision(
        "DENSE", "selected_independent_support", _snapshot_json(candidate["output"]),
        stats.avg_delta, stats.min_delta, stats.candidate_sheet_wins,
    )
