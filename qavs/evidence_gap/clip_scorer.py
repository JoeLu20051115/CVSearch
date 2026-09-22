"""Local, frozen CLIP scoring wrapper used by query-aware ranking."""

import math
from numbers import Real
from typing import Any

from PIL import Image


class ClipScorer:
    def __init__(self, device: str | None = None, *, model_path: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()

    @staticmethod
    def _validate_inputs(images: list[Image.Image], texts: list[str]) -> None:
        if not images or not texts:
            raise ValueError("images and texts must be nonempty")
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("images must be PIL images")
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("texts must be nonempty strings")

    @staticmethod
    def _finite(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError("CLIP scores must be finite")
        return float(value)

    def score(self, images: list[Image.Image], texts: list[str]) -> list[list[float]]:
        self._validate_inputs(images, texts)
        with self.torch.inference_mode():
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**batch)
            image_features = self.torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
            text_features = self.torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
            matrix = (image_features @ text_features.T).float().cpu().tolist()
        if len(matrix) != len(images) or any(len(row) != len(texts) for row in matrix):
            raise ValueError("CLIP returned an unexpected score shape")
        return [[self._finite(score) for score in row] for row in matrix]
