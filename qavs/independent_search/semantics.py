"""Semantic answer catalogs shared by Revision-3 verification and projection."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any

from PIL import Image

from qavs.evidence_gap.answers import canonical_text, parse_option_block
from qavs.evidence_gap.provenance import canonical_sha256
from qavs.evidence_gap.helpers import (
    UnprojectableSemanticOptionSet,
    build_semantic_option_set,
)


SUPPORT_LABELS = ("Support", "Refute", "Insufficient")
_OPTION_VERIFIER_PROMPT = (
    "Assess only direct visible evidence in this image; do not answer with another "
    "option and do not infer missing evidence. Classify with exactly one code: "
    "A = Support, B = Refute, C = Insufficient. Return only A, B, or C.\n"
    "Question: {question}\nCandidate answer: {option}\n"
    "Required visible evidence:\n{requirements}"
)


@dataclass(frozen=True)
class OptionEntry:
    key: str
    text: str
    output: Any


@dataclass(frozen=True)
class OptionCatalog:
    answer_type: str
    entries: tuple[OptionEntry, ...]
    identity_sha256: str

    def project(self, key: str) -> Any:
        matches = [entry for entry in self.entries if entry.key == key]
        if len(matches) != 1:
            raise KeyError(f"unknown semantic option key: {key!r}")
        output = matches[0].output
        if self.answer_type == "option_list":
            return list(output)
        return copy.deepcopy(output)


@dataclass(frozen=True)
class LabelDistribution:
    labels: tuple[str, str, str]
    losses: tuple[float, float, float]
    probabilities: tuple[float, float, float]
    winner: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "losses": list(self.losses),
            "probabilities": list(self.probabilities),
            "winner": self.winner,
        }


@dataclass(frozen=True)
class OptionSupport:
    key: str
    text_sha256: str
    distribution: LabelDistribution
    raw_support: float
    normalized_support: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "text_sha256": self.text_sha256,
            "distribution": self.distribution.to_dict(),
            "raw_support": self.raw_support,
            "normalized_support": self.normalized_support,
        }


@dataclass(frozen=True)
class OptionSupportVector:
    catalog_sha256: str
    options: tuple[OptionSupport, ...]
    valid: bool
    model_calls: int
    processed_pixels: int
    checkpoint_sha256: str
    prompt_template_sha256: str

    @property
    def top_key(self) -> str:
        if not self.options:
            raise ValueError("option support vector is empty")
        return max(
            enumerate(self.options),
            key=lambda item: (item[1].normalized_support, -item[0]),
        )[1].key

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_sha256": self.catalog_sha256,
            "options": [item.to_dict() for item in self.options],
            "valid": self.valid,
            "model_calls": self.model_calls,
            "processed_pixels": self.processed_pixels,
            "checkpoint_sha256": self.checkpoint_sha256,
            "prompt_template_sha256": self.prompt_template_sha256,
        }


def _visible_options(value: Any) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError("logits_match options must be a nonempty string sequence")
    options = tuple(" ".join(item.split()) for item in value)
    if len({canonical_text(item) for item in options}) != len(options):
        raise ValueError("semantic options must be unique after normalization")
    return options


def _surface_distinct_option_set(option_blocks: Any) -> dict[str, Any]:
    """Project the one HR case whose choices differ only by capitalization."""
    if (
        isinstance(option_blocks, (str, bytes))
        or not isinstance(option_blocks, Sequence)
        or len(option_blocks) != 4
    ):
        raise ValueError("HR option reconstruction requires four blocks")
    parsed = [parse_option_block(block) for block in option_blocks]
    if any(tuple(block) != ("A", "B", "C", "D") for block in parsed):
        raise ValueError("each HR block must contain A through D exactly once")
    choices = tuple(" ".join(value.split()) for value in parsed[0].values())
    if len(set(choices)) != 4:
        raise UnprojectableSemanticOptionSet(
            "HR choices must remain unique after whitespace normalization"
        )
    expected = set(choices)
    reverse = []
    for block in parsed:
        mapping = {" ".join(text.split()): letter for letter, text in block.items()}
        if len(mapping) != 4 or set(mapping) != expected:
            raise ValueError("HR blocks must contain the same surface choices")
        reverse.append(mapping)
    return {
        "choices": choices,
        "letters_by_choice": tuple(
            tuple(mapping[text] for mapping in reverse) for text in choices
        ),
    }


def build_option_catalog(policy: Mapping[str, Any]) -> OptionCatalog:
    if not isinstance(policy, Mapping):
        raise TypeError("policy must be a mapping")
    answer_type = policy.get("answer_type")
    if answer_type == "yes_no":
        entries = (
            OptionEntry(key="yes", text="Yes", output="yes"),
            OptionEntry(key="no", text="No", output="no"),
        )
    elif answer_type == "logits_match":
        options = _visible_options(policy.get("options"))
        entries = tuple(
            OptionEntry(key=f"index:{index}", text=text, output=index)
            for index, text in enumerate(options)
        )
    elif answer_type == "option_list":
        surface_keys = False
        try:
            material = build_semantic_option_set(policy.get("options"))
        except UnprojectableSemanticOptionSet:
            material = _surface_distinct_option_set(policy.get("options"))
            surface_keys = True
        keys = (
            tuple(f"surface:{index}" for index in range(4))
            if surface_keys else tuple(material["canonical_choices"])
        )
        entries = tuple(
            OptionEntry(
                key=key,
                text=text,
                output=tuple(material["letters_by_choice"][index]),
            )
            for index, (text, key) in enumerate(zip(
                material["choices"], keys, strict=True,
            ))
        )
    else:
        raise ValueError(
            "Revision-3 option catalogs support yes_no, logits_match, and option_list"
        )
    identity = canonical_sha256({
        "answer_type": answer_type,
        "entries": [
            {"key": item.key, "text": item.text, "output": item.output}
            for item in entries
        ],
    })
    return OptionCatalog(
        answer_type=str(answer_type), entries=entries, identity_sha256=identity,
    )


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not (normalized := " ".join(value.split())):
        raise ValueError(f"{name} must be nonempty text")
    return normalized


def verify_option_support(
    *,
    question: str,
    catalog: OptionCatalog,
    image: Image.Image,
    requirements: tuple[str, ...],
    conditional_losses: Callable[
        [Image.Image, str, tuple[str, str, str]],
        tuple[int, Sequence[float], int],
    ],
    checkpoint_sha256: str,
    generator_checkpoint_sha256: str,
) -> OptionSupportVector:
    """Compute one complete Support/Refute/Insufficient vector for a view."""
    question = _text(question, "question")
    if not isinstance(catalog, OptionCatalog) or not catalog.entries:
        raise TypeError("catalog must be a nonempty OptionCatalog")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if (
        not isinstance(requirements, tuple)
        or not requirements
        or any(not isinstance(item, str) or not item.strip() for item in requirements)
    ):
        raise ValueError("requirements must be a nonempty immutable text tuple")
    requirements = tuple(_text(item, "requirement") for item in requirements)
    if not callable(conditional_losses):
        raise TypeError("conditional_losses must be callable")
    checkpoint = _digest(checkpoint_sha256, "verifier checkpoint")
    generator_checkpoint = _digest(
        generator_checkpoint_sha256, "generator checkpoint"
    )
    if checkpoint == generator_checkpoint:
        raise ValueError("independent verifier must use a different checkpoint")

    rows: list[tuple[OptionEntry, LabelDistribution]] = []
    model_calls = 0
    requirement_text = "\n".join(
        f"- {index + 1}. {item}" for index, item in enumerate(requirements)
    )
    for entry in catalog.entries:
        prompt = _OPTION_VERIFIER_PROMPT.format(
            question=question, option=entry.text, requirements=requirement_text,
        )
        result = conditional_losses(image, prompt, SUPPORT_LABELS)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError(
                "conditional verifier must return winner, losses, and model calls"
            )
        winner, losses, calls = result
        if (
            isinstance(losses, (str, bytes))
            or not isinstance(losses, Sequence)
            or len(losses) != 3
        ):
            raise ValueError("conditional verifier must return three finite losses")
        values = tuple(float(item) for item in losses)
        if not all(math.isfinite(item) for item in values):
            raise ValueError("conditional verifier must return three finite losses")
        expected_winner = min(range(3), key=values.__getitem__)
        if type(winner) is not int or winner != expected_winner:
            raise ValueError("conditional verifier winner disagrees with losses")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
            raise ValueError("conditional verifier model calls must be positive")
        minimum = min(values)
        weights = tuple(math.exp(-(item - minimum)) for item in values)
        total = math.fsum(weights)
        probabilities = tuple(item / total for item in weights)
        rows.append((entry, LabelDistribution(
            labels=SUPPORT_LABELS,
            losses=values,
            probabilities=probabilities,
            winner=SUPPORT_LABELS[winner],
        )))
        model_calls += calls

    raw_support = [row.probabilities[0] for _, row in rows]
    total_support = math.fsum(raw_support)
    if not math.isfinite(total_support) or total_support <= 0.0:
        raise ValueError("option vector total raw support must be positive")
    options = tuple(
        OptionSupport(
            key=entry.key,
            text_sha256=hashlib.sha256(entry.text.encode("utf-8")).hexdigest(),
            distribution=row,
            raw_support=support,
            normalized_support=support / total_support,
        )
        for (entry, row), support in zip(rows, raw_support, strict=True)
    )
    return OptionSupportVector(
        catalog_sha256=catalog.identity_sha256,
        options=options,
        valid=any(
            item.distribution.winner != "Insufficient" for item in options
        ),
        model_calls=model_calls,
        processed_pixels=model_calls * image.width * image.height,
        checkpoint_sha256=checkpoint,
        prompt_template_sha256=hashlib.sha256(
            _OPTION_VERIFIER_PROMPT.encode("utf-8")
        ).hexdigest(),
    )
