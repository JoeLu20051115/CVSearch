"""Shared answer-free evidence-support contract for non-Qwen backbones."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Any, Callable

import PIL
from PIL import Image
import torch
import transformers

from cvsearch.evidence_gap.types import (
    EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE,
    EVIDENCE_SUPPORT_TRANSFORM,
    EvidenceRequirement,
    EvidenceSupportResult,
)


ANSWER_FREE_SUPPORT_PROMPT_VERSION = "qwen_answer_free_evidence_support_v1"
ANSWER_FREE_SUPPORT_PROCESSOR_MODE = (
    "single_rendered_view_chat_left_padding_final_yes_no_logits"
)
ANSWER_FREE_SUPPORT_TEMPLATE = (
    "Assess only whether the displayed visual observation provides sufficient visible "
    "evidence for the question and every ordered evidence requirement. Do not answer "
    "the question and do not infer missing evidence.\n"
    "Question: {question}\n"
    "Ordered evidence requirements:\n{requirements}\n"
    "Does this observation provide sufficient visible evidence for the complete "
    "requirement set? Answer Yes or No."
)
ANSWER_FREE_SUPPORT_TEMPLATE_SHA256 = hashlib.sha256(
    ANSWER_FREE_SUPPORT_TEMPLATE.encode("utf-8")
).hexdigest()


def _qualified_class(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def processor_fingerprint(wrapper: Any) -> tuple[dict[str, Any], str]:
    model = wrapper.model
    config = model.config
    tokenizer = wrapper.tokenizer
    checkpoint = os.path.realpath(wrapper.model_checkpoint)
    image_processor = getattr(wrapper, "image_processor", None)
    architectures = getattr(config, "architectures", None)
    payload = {
        "checkpoint_resolved": checkpoint,
        "checkpoint_snapshot": os.path.basename(checkpoint.rstrip(os.sep)),
        "model_config_class": _qualified_class(config),
        "model_type": getattr(config, "model_type", None),
        "architectures": None if architectures is None else list(architectures),
        "wrapper_dtype": str(wrapper.dtype),
        "model_dtype": str(getattr(model, "dtype", None)),
        "wrapper_device": str(wrapper.device),
        "model_device": str(getattr(model, "device", None)),
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "processor_class": None,
        "image_processor_class": (
            None if image_processor is None else _qualified_class(image_processor)
        ),
        "tokenizer_class": _qualified_class(tokenizer),
        "tokenizer_padding_side_default": getattr(tokenizer, "padding_side", None),
        "effective_padding_side": "left",
        "runtime_transformers_version": transformers.__version__,
        "runtime_torch_version": torch.__version__,
        "runtime_pillow_version": PIL.__version__,
        "checkpoint_transformers_version": getattr(config, "transformers_version", None),
    }
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload, checkpoint


def prepare_evidence_support(
    wrapper: Any,
    question: str,
    requirements: tuple[EvidenceRequirement, ...],
    prompt_builder: Callable[[str], str],
) -> dict[str, Any] | None:
    if not isinstance(requirements, tuple) or not all(
        isinstance(item, EvidenceRequirement) for item in requirements
    ):
        raise TypeError("requirements must be an immutable EvidenceRequirement tuple")
    if not requirements:
        return None
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty q0 string")
    yes_tokens = tuple(int(value) for value in wrapper.tokenizer("Yes").input_ids)
    no_tokens = tuple(int(value) for value in wrapper.tokenizer("No").input_ids)
    yes_id, no_id = int(wrapper.index_yes), int(wrapper.index_no)
    if yes_id == no_id or yes_id not in yes_tokens or no_id not in no_tokens:
        raise ValueError("Yes and No token provenance differs from wrapper initialization")
    fingerprint, checkpoint = processor_fingerprint(wrapper)
    requirement_lines = "\n".join(
        f"{index + 1}. [{item.requirement_id}] {item.text}"
        for index, item in enumerate(requirements)
    )
    support_text = ANSWER_FREE_SUPPORT_TEMPLATE.format(
        question=question, requirements=requirement_lines,
    )
    chat_prompt = prompt_builder(support_text)
    return {
        "yes_tokens": yes_tokens,
        "no_tokens": no_tokens,
        "yes_id": yes_id,
        "no_id": no_id,
        "fingerprint": fingerprint,
        "checkpoint": checkpoint,
        "chat_prompt": chat_prompt,
        "prompt_sha256": hashlib.sha256(chat_prompt.encode("utf-8")).hexdigest(),
        "prompt_template_sha256": ANSWER_FREE_SUPPORT_TEMPLATE_SHA256,
        "prompt_version": ANSWER_FREE_SUPPORT_PROMPT_VERSION,
        "processor_mode": ANSWER_FREE_SUPPORT_PROCESSOR_MODE,
        "p_yes_transform": EVIDENCE_SUPPORT_TRANSFORM,
    }


def validate_observation_identity(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("observation_identity must be canonical strict JSON")
    try:
        parsed = json.loads(value, parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"non-finite constant {item}")
        ))
        canonical = json.dumps(
            parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("observation_identity must be canonical strict JSON") from error
    if canonical != value:
        raise ValueError("observation_identity must be canonical strict JSON")


def finalize_evidence_support(
    prepared: dict[str, Any], requirements: tuple[EvidenceRequirement, ...],
    rendered_observation: Image.Image, observation_identity: str,
    pair: torch.Tensor, started: float,
) -> EvidenceSupportResult:
    if not isinstance(rendered_observation, Image.Image):
        raise TypeError("rendered_observation must be a PIL image")
    validate_observation_identity(observation_identity)
    if pair.numel() != 2:
        raise ValueError("support logits must contain exactly two values")
    probabilities = torch.softmax(pair, dim=-1)
    yes_logit, no_logit = (float(value) for value in pair.detach().cpu())
    p_yes, p_no = (float(value) for value in probabilities.detach().cpu())
    if not all(math.isfinite(value) for value in (yes_logit, no_logit, p_yes, p_no)):
        raise ValueError("support logits and probabilities must be finite")
    if abs((p_yes + p_no) - 1.0) > EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE:
        raise ValueError("support probabilities must be normalized")
    pixels = (
        rendered_observation.mode.encode("utf-8") + b"\0"
        + f"{rendered_observation.width}x{rendered_observation.height}".encode("ascii")
        + b"\0" + rendered_observation.tobytes()
    )
    return EvidenceSupportResult(
        requirements=requirements,
        requirement_set_id=EvidenceSupportResult.requirement_set_id_for(requirements),
        observation_identity=observation_identity,
        prompt_version=prepared["prompt_version"],
        prompt_template_sha256=prepared["prompt_template_sha256"],
        prompt_sha256=prepared["prompt_sha256"],
        processor_mode=prepared["processor_mode"],
        processor_fingerprint_json=json.dumps(
            prepared["fingerprint"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ),
        checkpoint=prepared["checkpoint"],
        yes_tokenization=prepared["yes_tokens"],
        no_tokenization=prepared["no_tokens"],
        yes_token_id=prepared["yes_id"], no_token_id=prepared["no_id"],
        p_yes_transform=prepared["p_yes_transform"],
        yes_logit=yes_logit, no_logit=no_logit, p_yes=p_yes, p_no=p_no,
        support_avg=p_yes, support_min=p_yes,
        observation_mode=rendered_observation.mode,
        observation_size=(rendered_observation.width, rendered_observation.height),
        view_sha256=hashlib.sha256(pixels).hexdigest(),
        elapsed_seconds=time.perf_counter() - started,
    )
