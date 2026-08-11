"""Runtime adapters for PDF-faithful query planning, trees, and uncertainty."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cvsearch.eval.phase7_uncertainty_confirmation import confirmation_prompt_material
from cvsearch.eval.phase10_paired_verifier import proposed_answer_text
from cvsearch.evidence_gap.answers import (
    aggregate_hr_answers,
    aggregate_vstar_losses,
    canonical_text,
    official_letter,
    parse_option_block,
)
from cvsearch.evidence_gap.input import sanitize_annotation
from cvsearch.evidence_gap.method import build_query_plan
from cvsearch.evidence_gap.search_state import SearchStateCollector
from cvsearch.evidence_gap.types import (
    AnswerRecord,
    EvidenceRequirement,
    sanitize_evidence_requirements,
)
from cvsearch.models.tree import NodeA, NodeState

from .pdf_controller import ActionOutcome, FrozenRankedQueue
from .pdf_controller import StateAssessment
from .pdf_types import (
    ActionName,
    CandidateDescriptor,
    EvidenceGapScores,
    SearchStateRecord,
)


_PLAN_PROMPT = (
    "Create answer-free visual search material for the question below. Return only one "
    "JSON object with exactly these keys: augmented_queries, evidence_items, "
    "global_scope_required. augmented_queries must contain four to six distinct short "
    "phrases for locating visible objects, details, or relations. Use at most two "
    "evidence_items, each in one of these exact forms: "
    "{{\"kind\":\"target_detail\",\"target\":\"visible subject\","
    "\"requirements\":[\"presence\",\"visual_detail\"]}}; "
    "{{\"kind\":\"relation_context\",\"targets\":[\"subject\",\"reference\"]}}; "
    "or {{\"kind\":\"coverage\",\"requirement\":\"global_scope\"}}. Do not invent "
    "new requirement values or schemas. Replace 'visible subject', 'subject', and "
    "'reference' with concrete answer-free noun phrases from the question; never copy "
    "those placeholders. global_scope_required is true only for count, absence, "
    "uniqueness, or all-object questions. Do not "
    "answer the question, mention candidate answers, or infer unseen details.\nQuestion: {question}"
)

_GAP_PROMPT = (
    "You are a visual tree-search ACTION SCORER. Do not answer the question. Do not "
    "describe evidence. Do not emit a list, prose, Markdown, or requirement IDs. Score "
    "only how much each next action is needed: zoom=missing local detail; "
    "split=need a finer child patch; expand=missing surrounding relation/context; "
    "next=current region is likely wrong or insufficient. Use independent numbers from "
    "0 to 1. Your entire response must be one single-line JSON object with exactly the "
    "four keys zoom, split, expand, next in that order and one numeric score per key. "
    "Choose the scores from the image; do not copy default values, and do not make all "
    "four scores identical.\n"
    "Question: {question}\nVisible requirements: {requirements}\n"
    "Return the four-number JSON object now."
)

_VERIFIER_PROMPT = (
    "Assess only direct visible support in the displayed observation. A non-root "
    "observation shows the whole-image overview above and the selected local detail "
    "below. The selected region is outlined in yellow in the overview; added context "
    "regions are outlined in cyan. Use both for identity, location, and relation. Do "
    "not replace the proposed answer and do not infer missing details.\nQuestion: {question}\n"
    "Proposed answer: {answer}\nRequired visible evidence: [{requirement_id}] "
    "{requirement}\nDoes this observation directly support the proposed answer for this "
    "specific requirement? Answer Yes or No."
)

_PAIRWISE_VERIFIER_PROMPT = (
    "Use only direct visible evidence in the displayed observation to answer the "
    "question. A non-root observation shows the whole-image overview above and the "
    "selected local detail below; yellow marks the selected region and cyan marks "
    "added context. Do not infer missing details.\nQuestion: {question}\n"
    "Required visible evidence: [{requirement_id}] {requirement}\n"
    "Complete the response with the candidate answer that is better supported."
)

_PAIRWISE_STAGED_PROMPT = (
    "{question}\nIdentify the visible {visual_property}, then answer with the "
    "supported candidate."
)

_ANSWER_FREE_DETAIL_TERMS = frozenset({
    "appearance", "color", "detail", "details", "material", "orientation",
    "presence", "shape", "size", "state", "text", "visual_detail",
})


def _normalized_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not (result := " ".join(value.split())):
        raise ValueError(f"{name} must be nonempty text")
    return result


def _requested_visual_property(question: str) -> str:
    """Map answer-free question wording to the visual property to inspect first."""
    text = _normalized_text(question, "question").casefold()
    if re.search(r"\b(colou?r|hue)\b", text):
        return "color"
    if re.search(r"\b(how many|count|number of)\b", text):
        return "count"
    if re.search(r"\b(read|text|word|written|says|label)\b", text):
        return "text"
    if re.search(r"\b(left|right|side|position|where|above|below|behind|front)\b", text):
        return "spatial relation"
    if re.search(r"\b(size|large|small|tall|short|wide|narrow)\b", text):
        return "size"
    return "attribute or relation"


def _strict_json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON") from error


@dataclass(frozen=True)
class PDFQueryPlan:
    main_query: str
    targets: tuple[str, ...]
    augmented_queries: tuple[str, ...]
    evidence_items: tuple[dict[str, Any], ...]
    global_scope_required: bool
    fallback_used: bool
    fallback_reason: str | None
    raw_response_sha256: str

    def __post_init__(self) -> None:
        _normalized_text(self.main_query, "main_query")
        for name, values in (("targets", self.targets), ("augmented_queries", self.augmented_queries)):
            if not isinstance(values, tuple) or not all(isinstance(item, str) and item for item in values):
                raise TypeError(f"{name} must be an immutable text tuple")
            if len({item.casefold() for item in values}) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if len(self.augmented_queries) < 3:
            raise ValueError("augmented_queries must retain at least three queries")
        if not isinstance(self.evidence_items, tuple) or not self.evidence_items:
            raise ValueError("evidence_items must be a nonempty immutable tuple")
        sanitize_evidence_requirements(self.evidence_items)
        if type(self.global_scope_required) is not bool or type(self.fallback_used) is not bool:
            raise TypeError("query-plan flags must be booleans")
        if self.fallback_used != (self.fallback_reason is not None):
            raise ValueError("fallback reason must exist exactly when fallback is used")
        if self.fallback_reason is not None:
            _normalized_text(self.fallback_reason, "fallback_reason")
        if not re.fullmatch(r"[0-9a-f]{64}", self.raw_response_sha256):
            raise ValueError("raw_response_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_query": self.main_query,
            "targets": list(self.targets),
            "augmented_queries": list(self.augmented_queries),
            "evidence_items": _strict_json_copy(self.evidence_items, "evidence_items"),
            "global_scope_required": self.global_scope_required,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "raw_response_sha256": self.raw_response_sha256,
        }


def _option_texts(policy: Mapping[str, Any]) -> tuple[str, ...]:
    options = policy["options"]
    if not isinstance(options, list):
        raise TypeError("options must be a list")
    if policy["answer_type"] == "logits_match":
        return tuple(_normalized_text(item, "option") for item in options)
    if policy["answer_type"] == "option_list":
        texts = []
        for block in options:
            texts.extend(parse_option_block(block).values())
        return tuple(dict.fromkeys(texts))
    return tuple(_normalized_text(item, "option") for item in options)


def _parse_plan(raw: str, policy: Mapping[str, Any], targets: tuple[str, ...]) -> PDFQueryPlan:
    if not isinstance(raw, str):
        raise TypeError("structured query plan response must be text")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("structured query plan response contains no JSON object")
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict) or set(data) != {
        "augmented_queries", "evidence_items", "global_scope_required",
    }:
        raise ValueError("structured query plan has an invalid exact schema")
    queries = data["augmented_queries"]
    if not isinstance(queries, list) or not 3 <= len(queries) <= 8:
        raise ValueError("structured query plan needs three to eight augmentations")
    normalized_queries = tuple(_normalized_text(item, "augmented query") for item in queries)
    if len({item.casefold() for item in normalized_queries}) != len(normalized_queries):
        raise ValueError("structured query augmentations must be distinct")
    if not isinstance(data["evidence_items"], list) or not data["evidence_items"]:
        raise ValueError("structured query plan needs evidence items")
    if type(data["global_scope_required"]) is not bool:
        raise TypeError("global_scope_required must be a boolean")
    expected_global_scope = build_query_plan(
        policy, targets,
    ).global_scope_required
    normalized_items = []
    for item in data["evidence_items"]:
        copied = _strict_json_copy(item, "evidence item")
        if isinstance(copied, dict) and copied.get("kind") == "target_detail":
            values = copied.get("requirements")
            normalized_values = (
                [] if not isinstance(values, list) else [
                    "_".join(str(value).casefold().replace("-", " ").split())
                    for value in values
                ]
            )
            if (
                normalized_values
                and len(normalized_values) <= 4
                and all(value in _ANSWER_FREE_DETAIL_TERMS for value in normalized_values)
            ):
                copied["requirements"] = ["presence", "visual_detail"]
        normalized_items.append(copied)
    if expected_global_scope:
        if not any(
            isinstance(item, dict) and item.get("kind") == "coverage"
            for item in normalized_items
        ):
            normalized_items.append({
                "kind": "coverage", "requirement": "global_scope",
            })
    else:
        normalized_items = [
            item for item in normalized_items
            if not (isinstance(item, dict) and item.get("kind") == "coverage")
        ]
    evidence_items = tuple(normalized_items)
    sanitize_evidence_requirements(evidence_items)
    placeholder_material = json.dumps(evidence_items, ensure_ascii=False).casefold()
    if any(
        re.search(rf"(?<!\w){re.escape(placeholder)}(?!\w)", placeholder_material)
        for placeholder in ("visible subject", "subject", "reference")
    ):
        raise ValueError("structured query plan retained schema placeholders")
    material = json.dumps(
        {"augmented_queries": normalized_queries, "evidence_items": evidence_items},
        ensure_ascii=False,
    ).casefold()
    for option in _option_texts(policy):
        normalized = option.casefold()
        if normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", material):
            raise ValueError("structured query plan leaked answer-option text")
    return PDFQueryPlan(
        main_query=policy["question"],
        targets=targets,
        augmented_queries=normalized_queries,
        evidence_items=evidence_items,
        global_scope_required=expected_global_scope,
        fallback_used=False,
        fallback_reason=None,
        raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


def _fallback_plan(
    policy: Mapping[str, Any], targets: tuple[str, ...], raw: str, reason: str,
) -> PDFQueryPlan:
    base = build_query_plan(policy, targets)
    subjects = list(targets)
    if not subjects:
        content = re.findall(r"[A-Za-z0-9]+", policy["question"])
        subjects = [" ".join(content[-6:]) or "question evidence"]
    queries = []
    for subject in subjects:
        queries.extend((
            f"locate {subject}",
            f"inspect visible details of {subject}",
            f"surrounding context and relations of {subject}",
        ))
    normalized = tuple(dict.fromkeys(" ".join(item.split()) for item in queries))
    while len(normalized) < 3:
        normalized += (f"visible evidence view {len(normalized) + 1}",)
    return PDFQueryPlan(
        main_query=policy["question"],
        targets=targets,
        augmented_queries=normalized,
        evidence_items=tuple(_strict_json_copy(item, "evidence item") for item in base.evidence_items),
        global_scope_required=base.global_scope_required,
        fallback_used=True,
        fallback_reason=reason,
        raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


def build_pdf_query_plan(
    policy_annotation: Mapping[str, Any],
    targets: Sequence[str],
    *,
    generator: Callable[[str], str],
) -> PDFQueryPlan:
    policy = sanitize_annotation(policy_annotation)
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence")
    normalized_targets = tuple(dict.fromkeys(
        _normalized_text(item, "target") for item in targets
    ))
    if not callable(generator):
        raise TypeError("generator must be callable")
    prompt = _PLAN_PROMPT.format(question=policy["question"])
    raw = generator(prompt)
    if not isinstance(raw, str):
        raise TypeError("generator must return text")
    try:
        return _parse_plan(raw, policy, normalized_targets)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        reason = f"{type(error).__name__}:{str(error)}"
        return _fallback_plan(policy, normalized_targets, raw, reason)


@dataclass(frozen=True)
class GapScoreResult:
    scores: EvidenceGapScores
    mode: str
    raw_response_sha256: str
    fallback_reason: str | None
    model_calls: int

    def __post_init__(self) -> None:
        if not isinstance(self.scores, EvidenceGapScores):
            raise TypeError("gap result scores must be EvidenceGapScores")
        if self.mode not in {"model_json", "analytic_fallback"}:
            raise ValueError("gap result mode is invalid")
        if self.mode == "model_json" and self.fallback_reason is not None:
            raise ValueError("model gap result cannot contain a fallback reason")
        if self.mode == "analytic_fallback" and not self.fallback_reason:
            raise ValueError("analytic gap result requires a fallback reason")
        if not re.fullmatch(r"[0-9a-f]{64}", self.raw_response_sha256):
            raise ValueError("gap raw-response digest is invalid")
        if self.model_calls != 1:
            raise ValueError("gap scorer must charge exactly one model call")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores.to_dict(),
            "mode": self.mode,
            "raw_response_sha256": self.raw_response_sha256,
            "fallback_reason": self.fallback_reason,
            "model_calls": self.model_calls,
        }


def _gap_mapping(value: Any, name: str) -> EvidenceGapScores:
    if not isinstance(value, Mapping) or set(value) != {"zoom", "split", "expand", "next"}:
        raise ValueError(f"{name} must contain exactly zoom, split, expand, and next")
    return EvidenceGapScores(
        zoom=value["zoom"], split=value["split"],
        expand=value["expand"], next=value["next"],
    )


def score_evidence_gaps(
    *,
    q0: str,
    requirements: tuple[EvidenceRequirement, ...],
    generator: Callable[[str], str],
    analytic: Mapping[str, float],
) -> GapScoreResult:
    question = _normalized_text(q0, "q0")
    if not isinstance(requirements, tuple) or not all(
        isinstance(item, EvidenceRequirement) for item in requirements
    ) or not requirements:
        raise ValueError("gap scorer requires an immutable nonempty requirement tuple")
    if not callable(generator):
        raise TypeError("gap generator must be callable")
    fallback_scores = _gap_mapping(analytic, "analytic gap scores")
    lines = "; ".join(item.text for item in requirements)
    prompt = _GAP_PROMPT.format(question=question, requirements=lines)
    try:
        raw = generator(prompt)
        if not isinstance(raw, str):
            raise TypeError("gap generator must return text")
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("gap response contains no JSON object")
        scores = _gap_mapping(json.loads(raw[start:end + 1]), "model gap scores")
        values = tuple(scores.to_dict().values())
        if max(values) == 0.0:
            raise ValueError("model gap scores are an all-zero non-decision")
    except Exception as error:
        if "raw" not in locals() or not isinstance(raw, str):
            raw = ""
        return GapScoreResult(
            scores=fallback_scores,
            mode="analytic_fallback",
            raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            fallback_reason=type(error).__name__,
            model_calls=1,
        )
    return GapScoreResult(
        scores=scores,
        mode="model_json",
        raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        fallback_reason=None,
        model_calls=1,
    )


@dataclass(frozen=True)
class IndependentSupportResult:
    per_requirement: tuple[float, ...]
    requirement_ids: tuple[str, ...]
    support_avg: float
    support_min: float
    independent: bool
    fallback_used: bool
    checkpoint_sha256: str
    failure_type: str | None
    failure_message_sha256: str | None
    model_calls: int

    def __post_init__(self) -> None:
        if not self.per_requirement or len(self.per_requirement) != len(self.requirement_ids):
            raise ValueError("support values must align with nonempty requirements")
        values = tuple(float(value) for value in self.per_requirement)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("support values must be finite probabilities")
        if abs(self.support_avg - math.fsum(values) / len(values)) > 1e-12:
            raise ValueError("support average does not match per-requirement values")
        if self.support_min != min(values):
            raise ValueError("support minimum does not match per-requirement values")
        if self.independent == self.fallback_used:
            raise ValueError("support must be either independent or fallback")
        if self.fallback_used != (self.failure_type is not None):
            raise ValueError("support failure type must match fallback status")
        if self.fallback_used != (self.failure_message_sha256 is not None):
            raise ValueError("support failure digest must match fallback status")
        for digest in (self.checkpoint_sha256, self.failure_message_sha256):
            if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("support digest is invalid")
        if isinstance(self.model_calls, bool) or not isinstance(self.model_calls, int) or self.model_calls < 0:
            raise ValueError("support model_calls must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_requirement": list(self.per_requirement),
            "requirement_ids": list(self.requirement_ids),
            "support_avg": self.support_avg,
            "support_min": self.support_min,
            "independent": self.independent,
            "fallback_used": self.fallback_used,
            "checkpoint_sha256": self.checkpoint_sha256,
            "failure_type": self.failure_type,
            "failure_message_sha256": self.failure_message_sha256,
            "model_calls": self.model_calls,
        }


@dataclass(frozen=True)
class PairwiseSupportResult:
    proposed: IndependentSupportResult
    reference: IndependentSupportResult
    mode: str
    model_calls: int
    paraphrase_ids: tuple[str, ...]
    proposed_by_requirement: tuple[tuple[float, ...], ...]
    reference_by_requirement: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.proposed, IndependentSupportResult) or not isinstance(
            self.reference, IndependentSupportResult
        ):
            raise TypeError("pairwise support requires two support results")
        if (
            self.proposed.requirement_ids != self.reference.requirement_ids
            or self.proposed.checkpoint_sha256 != self.reference.checkpoint_sha256
        ):
            raise ValueError("pairwise support must share requirements and checkpoint")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("pairwise support mode must be nonempty")
        if (
            isinstance(self.model_calls, bool) or not isinstance(self.model_calls, int)
            or self.model_calls <= 0
        ):
            raise ValueError("pairwise support model_calls must be positive")
        if self.paraphrase_ids != (
            "question", "requirement_conditioned", "coarse_to_fine",
        ):
            raise ValueError("pairwise support paraphrases are not the frozen set")
        if (
            len(self.proposed_by_requirement) != len(self.proposed.per_requirement)
            or len(self.reference_by_requirement) != len(self.reference.per_requirement)
        ):
            raise ValueError("pairwise paraphrase rows must align with requirements")
        for proposed_row, reference_row in zip(
            self.proposed_by_requirement, self.reference_by_requirement,
        ):
            if (
                len(proposed_row) != len(self.paraphrase_ids)
                or len(reference_row) != len(self.paraphrase_ids)
            ):
                raise ValueError("pairwise paraphrase rows have invalid width")
            if any(
                not math.isclose(proposed + reference, 1.0, rel_tol=0.0, abs_tol=1e-12)
                for proposed, reference in zip(proposed_row, reference_row)
            ):
                raise ValueError("pairwise paraphrase probabilities must be normalized")
        for proposed, reference in zip(
            self.proposed.per_requirement, self.reference.per_requirement,
        ):
            if not math.isclose(proposed + reference, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("pairwise requirement probabilities must be normalized")


def compare_paired_support(
    proposed: IndependentSupportResult,
    reference: IndependentSupportResult,
    *,
    min_avg_delta: float = 0.05,
    min_requirement_delta: float = 0.0,
) -> dict[str, Any]:
    """Calibrate an answer change against CVSearch on the identical visual view."""
    if not isinstance(proposed, IndependentSupportResult) or not isinstance(
        reference, IndependentSupportResult
    ):
        raise TypeError("paired support requires two independent support results")
    for value, name in (
        (min_avg_delta, "minimum average delta"),
        (min_requirement_delta, "minimum requirement delta"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if (
        proposed.requirement_ids != reference.requirement_ids
        or proposed.checkpoint_sha256 != reference.checkpoint_sha256
    ):
        raise ValueError("paired support must share requirements and verifier checkpoint")
    deltas = tuple(
        candidate - baseline
        for candidate, baseline in zip(
            proposed.per_requirement, reference.per_requirement,
        )
    )
    avg_delta = proposed.support_avg - reference.support_avg
    minimum_delta = min(deltas)
    wins = sum(delta > 0.0 for delta in deltas)
    eligible = (
        proposed.independent and reference.independent
        and not proposed.fallback_used and not reference.fallback_used
        and wins == len(deltas)
        and avg_delta > float(min_avg_delta)
        and minimum_delta >= float(min_requirement_delta)
    )
    return {
        "selected": eligible,
        "avg_delta": avg_delta,
        "min_delta": minimum_delta,
        "requirement_wins": wins,
        "requirement_count": len(deltas),
        "min_avg_delta": float(min_avg_delta),
        "min_requirement_delta": float(min_requirement_delta),
    }


def _support_probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("support callback must return a probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("support callback must return a probability")
    return result


def wrapper_pairwise_option_probabilities(
    model: Any,
    image: Image.Image,
    prompt: str,
    proposed_answer: str,
    reference_answer: str,
) -> tuple[float, float, int, str]:
    """Compare two answers directly with an independent model on one shared view."""
    if not isinstance(image, Image.Image):
        raise TypeError("pairwise verifier image must be a PIL image")
    prompt = _normalized_text(prompt, "pairwise verifier prompt")
    proposed_answer = _normalized_text(proposed_answer, "proposed answer")
    reference_answer = _normalized_text(reference_answer, "reference answer")
    if proposed_answer == reference_answer:
        raise ValueError("pairwise verifier answers must differ")
    conditional = getattr(model, "multiple_choices_with_losses", None)
    if callable(conditional):
        root = NodeA(NodeState(image, [0, 0, image.width, image.height]))
        root.is_root = True
        root.search_source = "global"
        result = conditional(
            image, prompt, [proposed_answer, reference_answer], [root],
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("conditional option verifier must return choice and losses")
        losses = result[1]
        if not isinstance(losses, Sequence) or len(losses) != 2:
            raise ValueError("conditional option verifier must return two losses")
        values = tuple(float(value) for value in losses)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("conditional option losses must be finite")
        minimum = min(values)
        weights = tuple(math.exp(-(value - minimum)) for value in values)
        total = math.fsum(weights)
        probabilities = tuple(value / total for value in weights)
        return probabilities[0], probabilities[1], 3, "conditional_option_loss"

    def proposition(answer: str) -> str:
        return (
            f"{prompt}\nProposed answer: {answer}\n"
            "Does the observation directly support this proposed answer? Answer Yes or No."
        )

    proposed = wrapper_yes_no_probability(model, image, proposition(proposed_answer))
    reference = wrapper_yes_no_probability(model, image, proposition(reference_answer))
    total = proposed + reference
    if total <= 0.0:
        return 0.5, 0.5, 2, "normalized_yes_no_fallback"
    return (
        proposed / total, reference / total, 2,
        "normalized_yes_no_fallback",
    )


def verify_paired_answer_support(
    *,
    q0: str,
    proposed_answer: str,
    reference_answer: str,
    requirements: tuple[EvidenceRequirement, ...],
    rendered_observation: Image.Image,
    pair_probability: Callable[[Image.Image, str, str, str], tuple[float, float, int, str]],
    checkpoint_sha256: str,
    generator_checkpoint_sha256: str,
) -> PairwiseSupportResult:
    """Score both answers contrastively for every evidence requirement."""
    question = _normalized_text(q0, "q0")
    proposed_answer = _normalized_text(proposed_answer, "proposed_answer")
    reference_answer = _normalized_text(reference_answer, "reference_answer")
    if proposed_answer == reference_answer:
        raise ValueError("paired answers must differ")
    if not isinstance(requirements, tuple) or not requirements or not all(
        isinstance(item, EvidenceRequirement) for item in requirements
    ):
        raise ValueError("paired verifier requires nonempty immutable evidence requirements")
    if not isinstance(rendered_observation, Image.Image):
        raise TypeError("rendered_observation must be PIL.Image")
    for digest, name in (
        (checkpoint_sha256, "verifier checkpoint"),
        (generator_checkpoint_sha256, "generator checkpoint"),
    ):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name} digest is invalid")
    if checkpoint_sha256 == generator_checkpoint_sha256:
        raise ValueError("independent verifier must use a different checkpoint")
    if not callable(pair_probability):
        raise TypeError("pair_probability must be callable")

    proposed_rows: list[tuple[float, ...]] = []
    reference_rows: list[tuple[float, ...]] = []
    modes: list[str] = []
    model_calls = 0
    for item in requirements:
        prompts = (
            question,
            _PAIRWISE_VERIFIER_PROMPT.format(
                question=question,
                requirement_id=item.requirement_id,
                requirement=item.text,
            ),
            _PAIRWISE_STAGED_PROMPT.format(
                question=question,
                visual_property=_requested_visual_property(question),
            ),
        )
        proposed_row: list[float] = []
        reference_row: list[float] = []
        for prompt in prompts:
            result = pair_probability(
                rendered_observation, prompt, proposed_answer, reference_answer,
            )
            if not isinstance(result, (tuple, list)) or len(result) != 4:
                raise ValueError("pair probability must return two probabilities, calls, and mode")
            proposed, reference = (
                _support_probability(result[0]), _support_probability(result[1]),
            )
            if not math.isclose(proposed + reference, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("pair probability output must be normalized")
            calls = result[2]
            mode = result[3]
            if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
                raise ValueError("pair probability model calls must be positive")
            if not isinstance(mode, str) or not mode:
                raise ValueError("pair probability mode must be nonempty")
            proposed_row.append(proposed)
            reference_row.append(reference)
            modes.append(mode)
            model_calls += calls
        proposed_rows.append(tuple(proposed_row))
        reference_rows.append(tuple(reference_row))
    if len(set(modes)) != 1:
        raise ValueError("pair probability mode must be consistent across requirements")

    proposed_values = [min(row) for row in proposed_rows]
    reference_values = [max(row) for row in reference_rows]

    requirement_ids = tuple(item.requirement_id for item in requirements)

    def support(values: list[float]) -> IndependentSupportResult:
        return IndependentSupportResult(
            per_requirement=tuple(values),
            requirement_ids=requirement_ids,
            support_avg=math.fsum(values) / len(values),
            support_min=min(values),
            independent=True,
            fallback_used=False,
            checkpoint_sha256=checkpoint_sha256,
            failure_type=None,
            failure_message_sha256=None,
            model_calls=0,
        )

    return PairwiseSupportResult(
        proposed=support(proposed_values),
        reference=support(reference_values),
        mode=f"worst_case_3x_{modes[0]}",
        model_calls=model_calls,
        paraphrase_ids=(
            "question", "requirement_conditioned", "coarse_to_fine",
        ),
        proposed_by_requirement=tuple(proposed_rows),
        reference_by_requirement=tuple(reference_rows),
    )


def verify_answer_support(
    *,
    q0: str,
    proposed_answer: str,
    requirements: tuple[EvidenceRequirement, ...],
    rendered_observation: Image.Image,
    probability: Callable[[Image.Image, str], float],
    checkpoint_sha256: str,
    generator_checkpoint_sha256: str,
    fallback_probability: Callable[[Image.Image, str], float] | None = None,
    coverage_fraction: float = 1.0,
) -> IndependentSupportResult:
    question = _normalized_text(q0, "q0")
    answer = _normalized_text(proposed_answer, "proposed_answer")
    if not isinstance(requirements, tuple) or not requirements or not all(
        isinstance(item, EvidenceRequirement) for item in requirements
    ):
        raise ValueError("verifier requires nonempty immutable evidence requirements")
    if not isinstance(rendered_observation, Image.Image):
        raise TypeError("rendered_observation must be PIL.Image")
    for digest, name in (
        (checkpoint_sha256, "verifier checkpoint"),
        (generator_checkpoint_sha256, "generator checkpoint"),
    ):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name} digest is invalid")
    if checkpoint_sha256 == generator_checkpoint_sha256:
        raise ValueError("independent verifier must use a different checkpoint")
    if not callable(probability):
        raise TypeError("probability must be callable")
    coverage = _support_probability(coverage_fraction)
    prompts = [
        _VERIFIER_PROMPT.format(
            question=question,
            answer=answer,
            requirement_id=item.requirement_id,
            requirement=item.text,
        )
        for item in requirements
    ]
    calls = 0
    failure: BaseException | None = None
    try:
        values = []
        for prompt in prompts:
            calls += 1
            values.append(_support_probability(probability(rendered_observation, prompt)))
    except Exception as error:
        failure = error
        fallback = fallback_probability or (lambda image, prompt: 0.0)
        if not callable(fallback):
            raise TypeError("fallback_probability must be callable")
        values = []
        for prompt in prompts:
            calls += 1
            values.append(_support_probability(fallback(rendered_observation, prompt)))
    values = [
        min(value, coverage) if item.kind == "coverage" else value
        for item, value in zip(requirements, values)
    ]
    average = math.fsum(values) / len(values)
    return IndependentSupportResult(
        per_requirement=tuple(values),
        requirement_ids=tuple(item.requirement_id for item in requirements),
        support_avg=average,
        support_min=min(values),
        independent=failure is None,
        fallback_used=failure is not None,
        checkpoint_sha256=checkpoint_sha256,
        failure_type=None if failure is None else type(failure).__name__,
        failure_message_sha256=(
            None if failure is None else hashlib.sha256(str(failure).encode()).hexdigest()
        ),
        model_calls=calls,
    )


def qwen_yes_no_probability(model: Any, image: Image.Image, prompt: str) -> float:
    """Run one final-token Yes/No probability on the independent Qwen wrapper."""
    import torch

    yes_tokens, no_tokens, yes_id, no_id = model._support_token_provenance()
    if yes_id == no_id or yes_id not in yes_tokens or no_id not in no_tokens:
        raise ValueError("Qwen Yes/No token provenance is invalid")
    chat_prompt = model.get_prompt_from_qs("<image>\n" + _normalized_text(prompt, "verifier prompt"))
    inputs = model.processor(
        text=[chat_prompt], images=[image], return_tensors="pt",
        padding=True, padding_side="left",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.model(**inputs)
    pair = outputs.logits[0, -1, [yes_id, no_id]]
    if pair.numel() != 2:
        raise ValueError("Qwen verifier logits must contain Yes and No")
    return _support_probability(float(torch.softmax(pair, dim=-1)[0].detach().cpu()))


def wrapper_yes_no_probability(model: Any, image: Image.Image, prompt: str) -> float:
    """Use the strongest available Yes/No confidence interface of a wrapper."""
    direct = getattr(model, "direct_yes_no_probability", None)
    if callable(direct):
        return _support_probability(direct(image, _normalized_text(prompt, "verifier prompt")))
    if callable(getattr(model, "_support_token_provenance", None)):
        return qwen_yes_no_probability(model, image, prompt)
    root = NodeA(NodeState(image, [0, 0, image.width, image.height]))
    root.is_root = True
    root.search_source = "global"
    confidence = model.get_confidence_value(
        [root], image, confidence_type="answering", input_ele=prompt,
    )
    value = float(confidence)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("verifier answering confidence must be finite in [-1, 1]")
    return (value + 1.0) / 2.0


def generate_text_only_response(model: Any, prompt: str) -> str:
    """Deterministic text-only generation shared by all supported wrappers."""
    prompt = _normalized_text(prompt, "text-only prompt")
    if callable(getattr(model, "generate_text_only", None)):
        raw = model.generate_text_only(prompt)
    else:
        chat_prompt = model.get_prompt_from_qs(prompt)
        inputs = model.processor(
            text=[chat_prompt], images=None, return_tensors="pt",
            padding=True, padding_side="left",
        ).to(model.device)
        generated = model.model.generate(
            **inputs, use_cache=True, max_new_tokens=256, do_sample=False,
        )
        trimmed = [
            output[len(input_ids):]
            for input_ids, output in zip(inputs.input_ids, generated)
        ]
        raw = model.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    if not isinstance(raw, str):
        raise TypeError("text-only model generation must return text")
    return raw


def analytic_gap_scores(
    adapter: "TreeActionAdapter", state: SearchStateRecord,
    requirements: tuple[EvidenceRequirement, ...],
) -> dict[str, float]:
    """Answer-free deterministic fallback using tree and coverage state only."""
    feasible = adapter.feasible(state)
    focus = adapter.catalog.node(state.path_keys[-1])
    image_area = adapter.image.width * adapter.image.height
    box_area = focus.descriptor.bbox_original[2] * focus.descriptor.bbox_original[3]
    area_fraction = min(1.0, max(0.0, box_area / image_area))
    has_relation = any(item.kind == "relation_context" for item in requirements)
    has_coverage = any(item.kind == "coverage" for item in requirements)
    return {
        "zoom": 0.0 if not feasible[ActionName.ZOOM] else min(1.0, 0.35 + 0.65 * (1.0 - area_fraction)),
        "split": 0.0 if not feasible[ActionName.SPLIT] else min(1.0, 0.55 + 0.45 * max(0.0, focus.complexity)),
        "expand": 0.0 if not feasible[ActionName.EXPAND] else (0.85 if (has_relation or has_coverage) and not state.context_keys else 0.35),
        "next": 0.0 if not feasible[ActionName.NEXT] else 0.55,
    }


class PDFStateEvaluator:
    """Re-answer, score gaps, and independently verify every new tree state."""

    def __init__(
        self,
        *,
        generator_model: Any,
        adapter: "TreeActionAdapter",
        policy_annotation: Mapping[str, Any],
        query_plan: PDFQueryPlan,
        verifier_probability: Callable[[Image.Image, str], float],
        verifier_checkpoint_sha256: str,
        generator_checkpoint_sha256: str,
        fallback_probability: Callable[[Image.Image, str], float] | None = None,
        pair_probability: Callable[
            [Image.Image, str, str, str], tuple[float, float, int, str]
        ] | None = None,
    ) -> None:
        self.generator_model = generator_model
        self.adapter = adapter
        self.policy = sanitize_annotation(policy_annotation)
        self.query_plan = query_plan
        self.requirements = sanitize_evidence_requirements(query_plan.evidence_items)
        self.verifier_probability = verifier_probability
        self.fallback_probability = fallback_probability
        self.pair_probability = pair_probability
        self.verifier_checkpoint_sha256 = verifier_checkpoint_sha256
        self.generator_checkpoint_sha256 = generator_checkpoint_sha256
        self.records: list[dict[str, Any]] = []
        self.states: dict[int, SearchStateRecord] = {}
        self.support_results: dict[int, IndependentSupportResult] = {}

    def estimate_support_cost(self, state: SearchStateRecord) -> tuple[int, int]:
        verifier_image = self.adapter.render_verifier_view(state)
        # A partial independent failure can be followed by one complete fallback pass.
        calls = 2 * len(self.requirements)
        return calls, calls * verifier_image.width * verifier_image.height

    def verify_output_support(
        self, state: SearchStateRecord, output: Any,
    ) -> IndependentSupportResult:
        verifier_image = self.adapter.render_verifier_view(state)
        answer_text = proposed_answer_text(
            self.policy["answer_type"], self.policy["options"], output,
        )
        if answer_text is None:
            answer_text = "unavailable semantic answer"
        return verify_answer_support(
            q0=self.policy["question"], proposed_answer=answer_text,
            requirements=self.requirements,
            rendered_observation=verifier_image,
            probability=self.verifier_probability,
            fallback_probability=self.fallback_probability,
            checkpoint_sha256=self.verifier_checkpoint_sha256,
            generator_checkpoint_sha256=self.generator_checkpoint_sha256,
            coverage_fraction=self.adapter.coverage_fraction(state),
        )

    def estimate_paired_support_cost(
        self, state: SearchStateRecord,
    ) -> tuple[int, int]:
        verifier_image = self.adapter.render_verifier_view(state)
        calls = 9 * len(self.requirements)
        return calls, calls * verifier_image.width * verifier_image.height

    def verify_paired_output_support(
        self, state: SearchStateRecord, proposed_output: Any, reference_output: Any,
    ) -> PairwiseSupportResult:
        if self.pair_probability is None:
            raise ValueError("paired answer calibration requires a pair probability callback")
        proposed_text = proposed_answer_text(
            self.policy["answer_type"], self.policy["options"], proposed_output,
        )
        reference_text = proposed_answer_text(
            self.policy["answer_type"], self.policy["options"], reference_output,
        )
        if proposed_text is None or reference_text is None:
            raise ValueError("paired answer calibration requires two semantic answers")
        return verify_paired_answer_support(
            q0=self.policy["question"],
            proposed_answer=proposed_text,
            reference_answer=reference_text,
            requirements=self.requirements,
            rendered_observation=self.adapter.render_verifier_view(state),
            pair_probability=self.pair_probability,
            checkpoint_sha256=self.verifier_checkpoint_sha256,
            generator_checkpoint_sha256=self.generator_checkpoint_sha256,
        )

    def estimate_cost(self, state: SearchStateRecord) -> tuple[int, int]:
        answer_image, _ = self.adapter.render_state(state)
        verifier_image = self.adapter.render_verifier_view(state)
        answer_calls = 3 if self.policy["answer_type"] == "logits_match" else 4
        # Reserve one independent attempt and a complete fallback attempt.
        support_calls = 2 * len(self.requirements)
        calls = answer_calls + 1 + support_calls
        pixels = (
            (answer_calls + 1) * answer_image.width * answer_image.height
            + support_calls * verifier_image.width * verifier_image.height
        )
        return calls, pixels

    def __call__(self, state: SearchStateRecord) -> StateAssessment:
        answer_image, answer_nodes = self.adapter.render_state(state)
        answer = answer_with_uncertainty(
            self.generator_model, self.policy, answer_image, answer_nodes,
        )
        gap = score_evidence_gaps(
            q0=self.policy["question"],
            requirements=self.requirements,
            generator=lambda prompt: self.generator_model.free_form_using_nodes(
                answer_image, prompt, answer_nodes,
            ),
            analytic=analytic_gap_scores(self.adapter, state, self.requirements),
        )
        verifier_image = self.adapter.render_verifier_view(state)
        support = self.verify_output_support(state, answer.output)
        answer_calls = 3 if self.policy["answer_type"] == "logits_match" else 4
        model_calls = answer_calls + gap.model_calls + support.model_calls
        processed_pixels = (
            (answer_calls + gap.model_calls) * answer_image.width * answer_image.height
            + support.model_calls * verifier_image.width * verifier_image.height
        )
        assessment = StateAssessment(
            answer=answer.output,
            uncertainty=answer.uncertainty,
            gaps=gap.scores,
            support_avg=support.support_avg,
            support_min=support.support_min,
            verifier_independent=support.independent,
            aggregation_available=answer.aggregation_available is not False,
            model_calls=model_calls,
            processed_pixels=processed_pixels,
        )
        self.records.append({
            "state": state.to_dict(),
            "answer": answer.to_dict(),
            "gap": gap.to_dict(),
            "support": support.to_dict(),
            "answer_view_size": [answer_image.width, answer_image.height],
            "verifier_view_size": [verifier_image.width, verifier_image.height],
            "assessment": {
                "uncertainty": assessment.uncertainty,
                "support_avg": assessment.support_avg,
                "support_min": assessment.support_min,
                "model_calls": assessment.model_calls,
                "processed_pixels": assessment.processed_pixels,
            },
        })
        self.states[state.state_id] = state
        self.support_results[state.state_id] = support
        _strict_json_copy(self.records[-1], "state evaluation trace")
        return assessment


@dataclass(frozen=True)
class CatalogNode:
    descriptor: CandidateDescriptor
    parent_key: str | None
    child_keys: tuple[str, ...]
    source: str | None
    complexity: float
    synthetic: bool = False


class TreeCatalog:
    """Immutable parent/child catalog backed by SearchStateCollector renderers."""

    def __init__(
        self,
        nodes: Mapping[str, CatalogNode],
        root_key: str,
        collector: SearchStateCollector,
        image: Image.Image,
        *,
        truncated_child_edges: int = 0,
    ) -> None:
        self._nodes = dict(nodes)
        self.root_key = root_key
        self._collector = collector
        self._image = image.copy()
        if (
            isinstance(truncated_child_edges, bool)
            or not isinstance(truncated_child_edges, int)
            or truncated_child_edges < 0
        ):
            raise ValueError("truncated_child_edges must be a non-negative integer")
        self.truncated_child_edges = truncated_child_edges
        if root_key not in self._nodes:
            raise ValueError("tree root is missing from catalog")
        for key in self._nodes:
            self.path_to(key)

    @classmethod
    def from_collector(cls, collector: SearchStateCollector, image: Image.Image) -> "TreeCatalog":
        if not isinstance(collector, SearchStateCollector):
            raise TypeError("collector must be SearchStateCollector")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be PIL.Image")
        payload = collector.to_dict()
        snapshots = payload["snapshots"]
        tree_snapshots = [item for item in snapshots if item["event"] == "tree_ready"]
        source_snapshots = tree_snapshots or [
            item for item in snapshots if item["event"] == "p0_selected"
        ]
        if not source_snapshots:
            raise ValueError("collector contains neither a full tree nor P0 candidates")

        raw_nodes: dict[str, dict[str, Any]] = {}
        group_by_key: dict[str, str] = {}
        for snapshot in source_snapshots:
            group_prefix = f"tree-{snapshot['search_call_ordinal']}"
            for index, raw in enumerate(snapshot["candidates"]):
                key = raw["canonical_key"]
                if key not in raw_nodes:
                    raw_nodes[key] = raw
                else:
                    existing = raw_nodes[key]
                    if (
                        existing.get("bbox_original") != raw.get("bbox_original")
                        or existing.get("parent_key") != raw.get("parent_key")
                        or existing.get("depth") != raw.get("depth")
                    ):
                        raise ValueError("duplicate tree key has inconsistent geometry")
                    existing["child_keys"] = list(dict.fromkeys(
                        list(existing.get("child_keys") or ())
                        + list(raw.get("child_keys") or ())
                    ))
                group_by_key.setdefault(
                    key, raw.get("parent_key") or f"{group_prefix}-root",
                )

        full_root = next((
            raw for raw in raw_nodes.values()
            if raw.get("depth") == 0
            and tuple(raw.get("bbox_original", ())) == (0, 0, image.width, image.height)
        ), None)

        # A second CVSearch pass can build a tree in a cropped image.  Its
        # emitted coordinates are already shifted back to the original image,
        # but its local root still has parent=None.  Treating that local root as
        # another global root makes an otherwise valid trace disconnected.
        # Graft each such root onto the smallest already-connected main-tree
        # region that contains it (normally the exact region CVSearch cropped).
        if full_root is not None:
            full_root_key = full_root["canonical_key"]

            def contains(outer: Mapping[str, Any], inner: Mapping[str, Any]) -> bool:
                ox, oy, ow, oh = (float(value) for value in outer["bbox_original"])
                ix, iy, iw, ih = (float(value) for value in inner["bbox_original"])
                return (
                    ox <= ix and oy <= iy
                    and ox + ow >= ix + iw and oy + oh >= iy + ih
                )

            def connected(key: str) -> bool:
                seen = set()
                current: str | None = key
                while current is not None and current not in seen:
                    if current == full_root_key:
                        return True
                    seen.add(current)
                    current = raw_nodes[current].get("parent_key")
                return False

            disconnected_roots = [
                key for key, raw in raw_nodes.items()
                if key != full_root_key and raw.get("parent_key") is None
            ]
            for key in disconnected_roots:
                raw = raw_nodes[key]
                anchors = [
                    (candidate_key, candidate)
                    for candidate_key, candidate in raw_nodes.items()
                    if candidate_key != key
                    and connected(candidate_key)
                    and contains(candidate, raw)
                ]
                if not anchors:
                    raise ValueError("cropped tree root is outside the main image tree")
                anchor_key, anchor = min(
                    anchors,
                    key=lambda item: (
                        float(item[1]["bbox_original"][2])
                        * float(item[1]["bbox_original"][3]),
                        -int(item[1].get("depth", 0)),
                        item[0],
                    ),
                )
                raw["parent_key"] = anchor_key
                anchor["child_keys"] = list(dict.fromkeys(
                    list(anchor.get("child_keys") or ()) + [key]
                ))

        synthetic_key = "pdf-root-" + hashlib.sha256(
            json.dumps({
                "mode": image.mode, "size": [image.width, image.height],
                "pixels": hashlib.sha256(image.tobytes()).hexdigest(),
            }, sort_keys=True).encode()
        ).hexdigest()
        root_key = full_root["canonical_key"] if full_root is not None else synthetic_key
        nodes: dict[str, CatalogNode] = {}
        truncated_child_edges = 0
        for native_ordinal, (key, raw) in enumerate(raw_nodes.items()):
            parent = raw.get("parent_key")
            if full_root is None and parent is None:
                parent = root_key
            declared_children = tuple(raw.get("child_keys") or ())
            children = tuple(child for child in declared_children if child in raw_nodes)
            truncated_child_edges += len(declared_children) - len(children)
            complexity = raw.get("complexity")
            if complexity is None:
                complexity = raw.get("prior_prob", 0.0)
            complexity = float(complexity)
            if not math.isfinite(complexity):
                raise ValueError("tree complexity must be finite")
            descriptor = CandidateDescriptor(
                canonical_key=key,
                sibling_group=parent or group_by_key[key],
                native_ordinal=native_ordinal,
                bbox_original=tuple(raw["bbox_original"]),
                depth=int(raw["depth"]),
                render_level=int(raw["render_level"]),
            )
            nodes[key] = CatalogNode(
                descriptor, parent, children, raw.get("source"), complexity,
            )
        if full_root is None:
            top_level = tuple(key for key, node in nodes.items() if node.parent_key == root_key)
            descriptor = CandidateDescriptor(
                canonical_key=root_key,
                sibling_group="synthetic-root",
                native_ordinal=0,
                bbox_original=(0.0, 0.0, float(image.width), float(image.height)),
                depth=0,
                render_level=0,
            )
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            complexity = float(pixels.var() / (255.0 ** 2))
            nodes[root_key] = CatalogNode(
                descriptor, None, top_level, "global", complexity, True,
            )
        for key, node in nodes.items():
            if node.parent_key is not None and node.parent_key not in nodes:
                raise ValueError(f"tree node {key} has a missing parent")
            if any(child not in nodes for child in node.child_keys):
                raise ValueError(f"tree node {key} has a missing child")
        return cls(
            nodes, root_key, collector, image,
            truncated_child_edges=truncated_child_edges,
        )

    def node(self, key: str) -> CatalogNode:
        try:
            return self._nodes[key]
        except KeyError as error:
            raise KeyError(f"unknown tree node: {key}") from error

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._nodes

    def children(self, key: str) -> tuple[str, ...]:
        return self.node(key).child_keys

    def path_to(self, key: str) -> tuple[str, ...]:
        path = []
        seen = set()
        current: str | None = key
        while current is not None:
            if current in seen:
                raise ValueError("tree catalog contains a parent cycle")
            seen.add(current)
            path.append(current)
            current = self.node(current).parent_key
        path.reverse()
        if not path or path[0] != self.root_key:
            raise ValueError("tree path does not begin at the catalog root")
        return tuple(path)

    def render_nodes(self, keys: Sequence[str]) -> list[NodeA]:
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
            raise TypeError("render keys must be a sequence")
        result = []
        for key in keys:
            item = self.node(key)
            if item.synthetic:
                node = NodeA(NodeState(self._image.copy(), list(item.descriptor.bbox_original)))
                node.search_source = "global"
            else:
                view = self._collector.support_view((key,))
                if view is None or len(view) != 1:
                    raise ValueError("tree node has no native render descriptor")
                node = view[0].render_node
            node.depth = item.descriptor.depth
            node.complexity = item.complexity
            node.is_root = key == self.root_key
            node.is_leaf = not item.child_keys
            result.append(node)
        return result


class TreeActionAdapter:
    """Bind the pure controller actions to a jointly ranked CVSearch tree."""

    def __init__(
        self,
        catalog: TreeCatalog,
        image: Image.Image,
        query_plan: PDFQueryPlan,
        ranker: Any,
        *,
        max_zoom_level: int = 2,
    ) -> None:
        if not isinstance(catalog, TreeCatalog):
            raise TypeError("catalog must be TreeCatalog")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be PIL.Image")
        if not isinstance(query_plan, PDFQueryPlan):
            raise TypeError("query_plan must be PDFQueryPlan")
        if not callable(ranker):
            raise TypeError("ranker must be callable")
        if isinstance(max_zoom_level, bool) or not isinstance(max_zoom_level, int) or max_zoom_level < 1:
            raise ValueError("max_zoom_level must be a positive integer")
        self.catalog = catalog
        self.image = image.copy()
        self.query_plan = query_plan
        self.ranker = ranker
        self.max_zoom_level = max_zoom_level
        self.queue = FrozenRankedQueue()
        self.ranking_details: list[dict[str, Any]] = []
        self._ranked_children: dict[str, tuple[str, ...]] = {}
        self.reveal_children(catalog.root_key)

    def reveal_children(self, parent_key: str) -> tuple[str, ...]:
        if parent_key in self._ranked_children:
            return self._ranked_children[parent_key]
        child_keys = self.catalog.children(parent_key)
        if not child_keys:
            self._ranked_children[parent_key] = ()
            return ()
        nodes = self.catalog.render_nodes(child_keys)
        result = self.ranker(
            nodes, self.image, self.query_plan.main_query,
            self.query_plan.augmented_queries,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("tree ranker must return ranked nodes and details")
        ranked_nodes, details = list(result[0]), list(result[1])
        if len(ranked_nodes) != len(nodes) or len(details) != len(nodes):
            raise ValueError("tree ranker must preserve every child")
        key_by_identity = {id(node): key for node, key in zip(nodes, child_keys)}
        if set(map(id, ranked_nodes)) != set(key_by_identity):
            raise ValueError("tree ranker changed the child identity set")
        ranked_keys = tuple(key_by_identity[id(node)] for node in ranked_nodes)
        score_by_key: dict[str, float] = {}
        enriched = []
        for key, detail in zip(ranked_keys, details):
            if not isinstance(detail, Mapping):
                raise TypeError("tree rank details must be mappings")
            score = detail.get("score")
            if not isinstance(score, Mapping) or "rank" not in score:
                raise ValueError("tree rank detail must contain the combined rank")
            rank = float(score["rank"])
            if not math.isfinite(rank) or not 0.0 <= rank <= 1.0:
                raise ValueError("tree combined rank must be finite in [0, 1]")
            score_by_key[key] = rank
            item = _strict_json_copy(dict(detail), "tree rank detail")
            item.update({"canonical_key": key, "parent_key": parent_key})
            enriched.append(item)
        self.queue.add_sibling_group(
            [self.catalog.node(key).descriptor for key in child_keys],
            ranked_keys, score_by_key,
        )
        self.ranking_details.extend(enriched)
        self._ranked_children[parent_key] = ranked_keys
        return ranked_keys

    @staticmethod
    def _visited(state: SearchStateRecord) -> set[str]:
        return set(state.visited_keys)

    def _zoom_level(self, state: SearchStateRecord) -> int:
        prefix = f"{state.path_keys[-1]}@zoom"
        levels = [
            int(item[len(prefix):]) for item in state.observation_keys
            if item.startswith(prefix) and item[len(prefix):].isdigit()
        ]
        return max(levels, default=0)

    @staticmethod
    def _outside_area(
        candidate: tuple[float, float, float, float],
        focus: tuple[float, float, float, float],
    ) -> float:
        cx, cy, cw, ch = candidate
        fx, fy, fw, fh = focus
        overlap_w = max(0.0, min(cx + cw, fx + fw) - max(cx, fx))
        overlap_h = max(0.0, min(cy + ch, fy + fh) - max(cy, fy))
        return cw * ch - overlap_w * overlap_h

    def _expand_candidate(self, state: SearchStateRecord) -> CandidateDescriptor | None:
        focus = self.catalog.node(state.path_keys[-1]).descriptor.bbox_original
        excluded = set(state.visited_keys) | set(state.context_keys) | {self.catalog.root_key}
        for candidate in self.queue.ranked_remaining(tuple(excluded)):
            if self._outside_area(candidate.bbox_original, focus) > 0.0:
                return candidate
        return None

    def _next_candidate(self, state: SearchStateRecord) -> CandidateDescriptor | None:
        excluded = set(state.visited_keys) | {self.catalog.root_key}
        remaining = self.queue.ranked_remaining(tuple(excluded))
        return None if not remaining else remaining[0]

    def feasible(self, state: SearchStateRecord) -> dict[ActionName, bool]:
        if not isinstance(state, SearchStateRecord):
            raise TypeError("state must be SearchStateRecord")
        focus = state.path_keys[-1]
        children = self.catalog.children(focus)
        has_unvisited_child = any(key not in self._visited(state) for key in children)
        return {
            ActionName.ZOOM: self._zoom_level(state) < self.max_zoom_level,
            ActionName.SPLIT: has_unvisited_child,
            # Finish the current root-to-leaf descent before lateral search.
            ActionName.EXPAND: (
                not has_unvisited_child and self._expand_candidate(state) is not None
            ),
            ActionName.NEXT: (
                not has_unvisited_child and self._next_candidate(state) is not None
            ),
        }

    @staticmethod
    def _unique(values: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))

    def execute(self, action: ActionName, state: SearchStateRecord) -> ActionOutcome:
        if action is ActionName.ZOOM:
            level = self._zoom_level(state) + 1
            if level > self.max_zoom_level:
                return ActionOutcome.no_op("maximum_zoom_level")
            return ActionOutcome.changed(
                path_keys=state.path_keys,
                focus_keys=state.focus_keys,
                context_keys=state.context_keys,
                visited_keys=state.visited_keys,
                observation_keys=state.observation_keys + (f"{state.path_keys[-1]}@zoom{level}",),
            )
        if action is ActionName.SPLIT:
            ranked = self.reveal_children(state.path_keys[-1])
            key = next((item for item in ranked if item not in self._visited(state)), None)
            if key is None:
                return ActionOutcome.no_op("leaf_or_children_visited")
            return ActionOutcome.changed(
                path_keys=state.path_keys + (key,),
                focus_keys=(key,),
                context_keys=state.context_keys,
                visited_keys=self._unique(state.visited_keys + (key,)),
                observation_keys=state.observation_keys + (f"{key}@base",),
            )
        if action is ActionName.EXPAND:
            candidate = self._expand_candidate(state)
            if candidate is None:
                return ActionOutcome.no_op("no_ranked_spatial_context")
            key = candidate.canonical_key
            return ActionOutcome.changed(
                path_keys=state.path_keys,
                focus_keys=state.focus_keys,
                context_keys=self._unique(state.context_keys + (key,)),
                visited_keys=self._unique(state.visited_keys + (key,)),
                observation_keys=state.observation_keys + (f"context:{key}",),
            )
        if action is ActionName.NEXT:
            candidate = self._next_candidate(state)
            if candidate is None:
                return ActionOutcome.no_op("ranked_queue_empty")
            key = candidate.canonical_key
            return ActionOutcome.changed(
                path_keys=self.catalog.path_to(key),
                focus_keys=(key,),
                context_keys=(),
                visited_keys=self._unique(state.visited_keys + (key,)),
                observation_keys=state.observation_keys + (f"{key}@base",),
            )
        raise ValueError("TreeActionAdapter executes only evidence-gap actions")

    def branch_available(self, state: SearchStateRecord) -> bool:
        visited = self._visited(state)
        return any(
            any(child not in visited for child in self.catalog.children(ancestor))
            for ancestor in state.path_keys
        )

    def render_state(self, state: SearchStateRecord) -> tuple[Image.Image, list[NodeA]]:
        level = self._zoom_level(state)
        if level:
            x, y, width, height = self.catalog.node(
                state.path_keys[-1]
            ).descriptor.bbox_original
            crop = self.image.crop((x, y, x + width, y + height))
            scale = 2 ** level
            size = (
                min(4096, max(1, round(crop.width * scale))),
                min(4096, max(1, round(crop.height * scale))),
            )
            return crop.resize(size, Image.Resampling.BICUBIC), []
        keys = self._unique(state.focus_keys + state.context_keys)
        return self.image.copy(), self.catalog.render_nodes(keys)

    def render_verifier_view(self, state: SearchStateRecord) -> Image.Image:
        level = self._zoom_level(state)
        if state.path_keys[-1] == self.catalog.root_key:
            return self.render_state(state)[0] if level else self.image.copy()
        if level:
            detail = self.render_state(state)[0]
        else:
            keys = self._unique(state.focus_keys + state.context_keys)
            boxes = [self.catalog.node(key).descriptor.bbox_original for key in keys]
            left = max(0, math.floor(min(box[0] for box in boxes)))
            top = max(0, math.floor(min(box[1] for box in boxes)))
            right = min(self.image.width, math.ceil(max(box[0] + box[2] for box in boxes)))
            bottom = min(self.image.height, math.ceil(max(box[1] + box[3] for box in boxes)))
            if right <= left or bottom <= top:
                raise ValueError("verifier view has empty focus/context union")
            detail = self.image.crop((left, top, right, bottom))

        def fit(view: Image.Image, max_edge: int) -> Image.Image:
            edge = max(view.size)
            if edge <= max_edge:
                return view
            scale = max_edge / edge
            return view.resize(
                (max(1, round(view.width * scale)), max(1, round(view.height * scale))),
                Image.Resampling.BICUBIC,
            )

        overview = fit(self.image.copy(), 1024)
        draw = ImageDraw.Draw(overview)
        scale_x = overview.width / self.image.width
        scale_y = overview.height / self.image.height
        line_width = max(2, round(min(overview.size) / 256))

        def outline(key: str, color: tuple[int, int, int]) -> None:
            x, y, width, height = self.catalog.node(key).descriptor.bbox_original
            draw.rectangle(
                (
                    max(0, math.floor(x * scale_x)),
                    max(0, math.floor(y * scale_y)),
                    min(overview.width - 1, math.ceil((x + width) * scale_x) - 1),
                    min(overview.height - 1, math.ceil((y + height) * scale_y) - 1),
                ),
                outline=color, width=line_width,
            )

        for key in state.context_keys:
            outline(key, (0, 255, 255))
        for key in state.focus_keys:
            outline(key, (255, 215, 0))
        detail = fit(detail, 2048)
        separator = 8
        width = max(overview.width, detail.width)
        canvas = Image.new(
            "RGB", (width, overview.height + separator + detail.height),
            (122, 116, 104),
        )
        canvas.paste(overview, ((width - overview.width) // 2, 0))
        canvas.paste(
            detail,
            ((width - detail.width) // 2, overview.height + separator),
        )
        return canvas

    @staticmethod
    def _union_area(boxes: Sequence[tuple[float, float, float, float]]) -> float:
        if not boxes:
            return 0.0
        xs = sorted({x for box in boxes for x in (box[0], box[0] + box[2])})
        area = 0.0
        for x0, x1 in zip(xs, xs[1:]):
            if x1 <= x0:
                continue
            intervals = sorted(
                (box[1], box[1] + box[3])
                for box in boxes
                if box[0] < x1 and box[0] + box[2] > x0
            )
            covered = 0.0
            if intervals:
                start, end = intervals[0]
                for next_start, next_end in intervals[1:]:
                    if next_start > end:
                        covered += end - start
                        start, end = next_start, next_end
                    else:
                        end = max(end, next_end)
                covered += end - start
            area += (x1 - x0) * covered
        return area

    def coverage_fraction(self, state: SearchStateRecord) -> float:
        inspected = [
            self.catalog.node(key).descriptor.bbox_original
            for key in state.visited_keys
            if key != self.catalog.root_key and key in self.catalog
        ]
        area = self._union_area(inspected)
        return min(1.0, max(0.0, area / (self.image.width * self.image.height)))


def absolute_location_constraints(question: str) -> tuple[str, ...]:
    """Extract answer-free absolute half-image constraints from a question."""
    question = _normalized_text(question, "question")
    image_terms = r"(?:image|picture|photo|frame)"
    constraints = []
    for name, term in (
        ("left", r"left(?:-hand)?"),
        ("right", r"right(?:-hand)?"),
        ("top", r"top|upper"),
        ("bottom", r"bottom|lower"),
    ):
        if re.search(
            rf"\b(?:{term})(?:\s+(?:side|half|edge|corner))?\s+of\s+"
            rf"(?:the\s+)?{image_terms}\b",
            question,
            flags=re.IGNORECASE,
        ):
            constraints.append(name)
    return tuple(constraints)


def absolute_location_geometry(
    adapter: TreeActionAdapter,
    state: SearchStateRecord,
    question: str,
    requirements: tuple[EvidenceRequirement, ...] = (),
) -> dict[str, Any]:
    """Check answer-free location and relation evidence against the proposal state."""
    if not isinstance(adapter, TreeActionAdapter):
        raise TypeError("absolute location geometry requires a tree adapter")
    if not isinstance(state, SearchStateRecord):
        raise TypeError("absolute location geometry requires a search state")
    if not isinstance(requirements, tuple) or not all(
        isinstance(item, EvidenceRequirement) for item in requirements
    ):
        raise TypeError("proposal geometry requires immutable evidence requirements")
    constraints = absolute_location_constraints(question)
    x, y, width, height = adapter.catalog.node(
        state.path_keys[-1]
    ).descriptor.bbox_original
    center_x = (x + width / 2.0) / adapter.image.width
    center_y = (y + height / 2.0) / adapter.image.height
    absolute_eligible = all({
        "left": center_x <= 0.5,
        "right": center_x >= 0.5,
        "top": center_y <= 0.5,
        "bottom": center_y >= 0.5,
    }[name] for name in constraints)
    relation_required = any(item.kind == "relation_context" for item in requirements)
    is_root = state.path_keys[-1] == adapter.catalog.root_key
    zoom_level = adapter._zoom_level(state)
    relation_enriched = (
        not relation_required or is_root or zoom_level > 0 or bool(state.context_keys)
    )
    return {
        "constraints": list(constraints),
        "focus_bbox": [x, y, width, height],
        "focus_center_fraction": [center_x, center_y],
        "absolute_eligible": absolute_eligible,
        "relation_context_required": relation_required,
        "relation_enriched": relation_enriched,
        "zoom_level": zoom_level,
        "context_count": len(state.context_keys),
        "eligible": absolute_eligible and relation_enriched,
    }


def _combined_uncertainty(
    record: AnswerRecord,
    prompt_winners: Sequence[Any],
    *,
    agreement: float | None = None,
) -> AnswerRecord:
    winners = list(prompt_winners)
    counts: dict[Any, int] = {}
    for winner in winners:
        counts[winner] = counts.get(winner, 0) + 1
    frequency = (
        max(counts.values()) / len(winners) if agreement is None and winners
        else record.frequency if agreement is None else agreement
    )
    if record.aggregation_available is False:
        confidence = 0.0
    else:
        confidence = (record.margin + frequency) / 2.0
    groups = dict(record.groups)
    groups["prompt_winners"] = winners
    groups["uncertainty_formula"] = "1-mean(normalized_margin,prompt_agreement)"
    return AnswerRecord(
        output=record.output,
        canonical_answer=record.canonical_answer,
        raw_outputs=record.raw_outputs,
        groups=groups,
        frequency=frequency,
        margin=record.margin,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        losses=record.losses,
        selected_from="pdf_state",
        aggregation_available=(
            True if record.aggregation_available is None else record.aggregation_available
        ),
        aggregation_reason=record.aggregation_reason,
    )


def answer_with_uncertainty(
    model: Any,
    policy_annotation: Mapping[str, Any],
    image: Image.Image,
    searched_nodes: Sequence[Any],
) -> AnswerRecord:
    policy = sanitize_annotation(policy_annotation)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image")
    nodes = list(searched_nodes)
    material = confirmation_prompt_material(
        policy["answer_type"], policy["question"], policy["options"],
    )
    if policy["answer_type"] == "logits_match":
        rows = []
        winners = []
        for prompt in material["prompts"]:
            winner, losses = model.multiple_choices_with_losses(
                image, prompt, policy["options"], nodes,
            )
            row = [float(value) for value in losses]
            rows.append(row)
            winners.append(int(winner))
        return _combined_uncertainty(aggregate_vstar_losses(rows), winners)
    if policy["answer_type"] == "option_list":
        outputs = [
            model.free_form_using_nodes(image, prompt, nodes)
            for prompt in material["prompts"]
        ]
        record = aggregate_hr_answers(policy["options"], outputs)
        semantic_winners = []
        for block, raw in zip(policy["options"], outputs):
            letter = official_letter(raw)
            options = parse_option_block(block)
            semantic_winners.append(
                None if letter not in options else canonical_text(options[letter])
            )
        return _combined_uncertainty(
            record, semantic_winners, agreement=record.frequency,
        )
    raise ValueError("PDF uncertainty supports only V* and HR-Bench answer types")
