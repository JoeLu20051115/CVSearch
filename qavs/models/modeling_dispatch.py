"""Minimal model-family dispatch shared by cross-backbone runners."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


def detect_model_family(model_path: str | Path) -> str:
    checkpoint = Path(model_path)
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise ValueError(f"checkpoint config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "")).casefold()
    architectures = " ".join(
        str(value) for value in config.get("architectures", ())
    ).casefold()
    identity = f"{checkpoint.name.casefold()} {model_type} {architectures}"
    if "llava" in identity:
        return "llava"
    if "internvl" in identity:
        return "internvl"
    if "qwen" in identity and "vl" in identity:
        return "qwen"
    raise ValueError(f"unsupported multimodal backbone: {model_type or architectures!r}")


def finalize_option_losses(losses: Sequence[Any]) -> tuple[int, list[float]]:
    if not losses:
        raise ValueError("option losses must be nonempty")
    values = [float(loss.detach().cpu()) for loss in losses]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("option losses must be finite")
    return min(range(len(values)), key=values.__getitem__), values


def single_token_choice_ids(tokenizer: Any, choices: Sequence[str]) -> tuple[int, ...]:
    """Resolve a frozen set of equal-footing one-token verifier codes."""
    if (
        isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence)
        or not choices or any(not isinstance(choice, str) or not choice for choice in choices)
    ):
        raise ValueError("choices must be a nonempty sequence of text codes")
    token_ids = []
    for choice in choices:
        encoded = tokenizer(choice, add_special_tokens=False)
        ids = getattr(encoded, "input_ids", None)
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or len(ids) != 1:
            raise ValueError("verifier codes must each map to one distinct token")
        token_id = ids[0]
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("verifier codes must each map to one distinct token")
        token_ids.append(token_id)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("verifier codes must each map to one distinct token")
    return tuple(token_ids)


def finalize_token_logits(logits: Any, token_ids: Sequence[int]) -> tuple[int, list[float]]:
    """Convert next-token logits into relative losses for an exact label softmax."""
    values = [-float(logits[token_id].detach().float().cpu()) for token_id in token_ids]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("label token logits must be nonempty and finite")
    return min(range(len(values)), key=values.__getitem__), values


def load_search_model(model_path: str | Path, device: str = "cuda:0") -> Any:
    import torch

    checkpoint = Path(model_path)
    family = detect_model_family(checkpoint)
    if family == "llava":
        from .modeling_llava import ModelGlobalLocal, ModelLocal

        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if "anyres" in str(config.get("image_aspect_ratio", "")).casefold():
            return ModelGlobalLocal(
                model_path=str(checkpoint), conv_type="qwen_1_5", device=device,
                patch_scale=1.2, bias_value=0.6,
            )
        return ModelLocal(
            model_path=str(checkpoint), conv_type="v1", device=device,
            patch_scale=None, bias_value=0.2,
        )
    if family == "internvl":
        from .modeling_internvl import ModelInternvl

        return ModelInternvl(
            model_path=str(checkpoint), device=device, torch_dtype=torch.bfloat16,
            patch_scale=1.2,
        )
    from .modeling_qwenvl import ModelQwenVL

    kwargs = {"load_in_8bit": True} if "32b" in str(checkpoint).casefold() else {}
    return ModelQwenVL(
        model_path=str(checkpoint), device=device, torch_dtype=torch.bfloat16,
        patch_scale=1.2, **kwargs,
    )
