import hashlib
from typing import Any, Literal, Mapping


POLICY_FIELDS = ("question", "options", "answer_type", "input_image")


def sanitize_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in POLICY_FIELDS if key not in annotation]
    if missing:
        raise ValueError(f"missing policy fields: {missing}")
    return {key: annotation[key] for key in POLICY_FIELDS}


def split_bucket(
    benchmark: str, input_image: str, seed: int = 260809
) -> Literal["dev", "holdout"]:
    digest = hashlib.sha256(f"{seed}:{benchmark}:{input_image}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") % 5 == 0 else "holdout"
