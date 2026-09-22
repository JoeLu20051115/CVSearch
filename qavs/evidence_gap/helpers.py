"""Pure prompt, answer, and query helpers used by QAVS inference."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from qavs.evidence_gap.answers import (
    aggregate_hr_answers,
    canonical_text,
    parse_option_block,
)
from qavs.evidence_gap.input import POLICY_FIELDS
from qavs.evidence_gap.provenance import canonical_sha256
from qavs.evidence_gap.types import QueryPlan


VSTAR_CONFIRMATION_TEMPLATES = (
    "{question}",
    "Answer this visual multiple-choice question: {question}",
    "Using only visible evidence, answer this multiple-choice question: {question}",
)


def confirmation_prompt_material(
    answer_type: str, question: str, options: Sequence[str],
) -> dict[str, Any]:
    """Return the frozen answer-free prompt list for one confirmation view."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("confirmation question must be nonempty")
    if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
        raise TypeError("confirmation options must be a sequence")
    frozen_options = tuple(options)
    if not frozen_options or not all(
        isinstance(option, str) and option for option in frozen_options
    ):
        raise ValueError("confirmation options must be nonempty strings")
    if answer_type == "logits_match":
        prompts = [
            template.format(question=question)
            for template in VSTAR_CONFIRMATION_TEMPLATES
        ]
    elif answer_type == "option_list":
        if len(frozen_options) != 4:
            raise ValueError("HR confirmation requires four option shuffles")
        prompts = [
            question + "\n" + option + "Answer the option letter directly."
            for option in frozen_options
        ]
    else:
        raise ValueError("confirmation answer type is unsupported")
    return {
        "answer_type": answer_type,
        "prompts": prompts,
        "prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in prompts
        ],
        "options_sha256": canonical_sha256(list(frozen_options)),
    }


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


class UnprojectableSemanticOptionSet(ValueError):
    """The option schema is valid but cannot be uniquely projected."""


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
        raise UnprojectableSemanticOptionSet(
            "first HR block contains duplicate semantic choices"
        )
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


_GLOBAL_SCOPE = re.compile(
    r"\b(how many|count|total count|throughout|all|every|none|no|without|absent|only|unique|each)\b",
    re.IGNORECASE,
)


_NUMBER_OF_SCOPE = re.compile(r"\bnumber of\b", re.IGNORECASE)


_IDENTIFIER_NUMBER = re.compile(
    r"\b(?:license(?:\s+plate)?|plate|registration|model|serial|"
    r"identification|id|route|flight|train|jersey|phone|telephone)\s+number\b",
    re.IGNORECASE,
)


_RELATION = re.compile(
    r"\b(beside|between|behind|in front of|left of|right of|side of|relative to|near|next to|above|below)\b",
    re.IGNORECASE,
)


def _global_scope_requested(text: str) -> bool:
    return bool(
        _GLOBAL_SCOPE.search(text)
        or (
            _NUMBER_OF_SCOPE.search(text)
            and not _IDENTIFIER_NUMBER.search(text)
        )
    )


def _strict_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON-safe") from error


def build_query_plan(policy_annotation: Mapping[str, Any], targets: Sequence[str]) -> QueryPlan:
    """Build the deterministic, answer-free fallback query plan."""
    if not isinstance(policy_annotation, Mapping):
        raise TypeError("policy_annotation must be a mapping")
    if set(policy_annotation) != set(POLICY_FIELDS):
        raise ValueError(f"policy_annotation must contain exactly {list(POLICY_FIELDS)}")
    question = policy_annotation["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of strings")
    normalized_targets: list[str] = []
    seen: set[str] = set()
    for value in targets:
        if not isinstance(value, str):
            raise TypeError("targets must contain only strings")
        target = " ".join(value.split())
        if not target:
            continue
        key = target.casefold()
        if key not in seen:
            normalized_targets.append(target)
            seen.add(key)

    augmented = tuple(f"locate and inspect {target}" for target in normalized_targets)
    evidence_items: list[dict[str, Any]] = [
        {"kind": "target_detail", "target": target, "requirements": ["presence", "visual_detail"]}
        for target in normalized_targets
    ]
    if normalized_targets and (len(normalized_targets) > 1 or _RELATION.search(question)):
        evidence_items.append({"kind": "relation_context", "targets": list(normalized_targets)})
    global_scope_required = _global_scope_requested(question)
    if global_scope_required:
        evidence_items.append({"kind": "coverage", "requirement": "global_scope"})
    if not evidence_items:
        evidence_items.append({"kind": "question_evidence", "requirement": "visual_detail"})

    plan = QueryPlan(
        main_query=question,
        targets=tuple(normalized_targets),
        augmented_queries=augmented,
        evidence_items=tuple(evidence_items),
        global_scope_required=global_scope_required,
        fallback_used=True,
    )
    _strict_json(plan.to_dict(), "query plan")
    return plan
