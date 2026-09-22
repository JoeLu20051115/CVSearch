"""Deterministic identities for visual evidence and answer options."""

import hashlib
import json
from typing import Any


def _strict_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json(value)).hexdigest()
