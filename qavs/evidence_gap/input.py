"""Only task inputs enter the visual search policy."""

from typing import Any, Mapping

POLICY_FIELDS = ("question", "options", "answer_type", "input_image")


def sanitize_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in POLICY_FIELDS if key not in annotation]
    if missing:
        raise ValueError(f"missing policy fields: {missing}")
    return {key: annotation[key] for key in POLICY_FIELDS}
