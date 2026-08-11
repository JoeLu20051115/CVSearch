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
