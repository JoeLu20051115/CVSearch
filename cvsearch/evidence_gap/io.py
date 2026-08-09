"""Crash-safe, ordered JSONL checkpoint output."""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import fcntl


_ORDINAL_KEY = "_eg_ordinal"
_FINGERPRINT_KEY = "_eg_run_fingerprint"
_RESERVED_KEYS = frozenset({_ORDINAL_KEY, _FINGERPRINT_KEY})
_DEFAULT_FINGERPRINT = "unspecified"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(raw_line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw_line.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("corrupt JSONL checkpoint row") from error
    if not isinstance(value, dict):
        raise TypeError("checkpoint rows must be JSON objects")
    # JSON accepts overflowing exponent syntax (for example 1e999) as inf.
    json.dumps(value, ensure_ascii=False, allow_nan=False)
    return value


def _normalize_ordinals(expected_ordinals: Any) -> tuple[int, ...]:
    if isinstance(expected_ordinals, (set, frozenset, Mapping, str, bytes)):
        raise TypeError("expected_ordinals must preserve order")
    try:
        values = tuple(expected_ordinals)
    except TypeError as error:
        raise TypeError("expected_ordinals must be an ordered iterable") from error
    if not values:
        raise ValueError("expected_ordinals must not be empty")

    normalized: list[int] = []
    for ordinal in values:
        if isinstance(ordinal, bool) or not isinstance(ordinal, Integral):
            raise TypeError("ordinals must be non-boolean integers")
        ordinal = int(ordinal)
        if ordinal < 0:
            raise ValueError("ordinals must be non-negative")
        normalized.append(ordinal)
    if len(set(normalized)) != len(normalized):
        raise ValueError("expected_ordinals must be unique")
    return tuple(normalized)


class JsonlCheckpointWriter:
    """Append a strict expected prefix and atomically publish it when complete."""

    def __init__(
        self,
        final_path: str | os.PathLike[str],
        expected_ordinals: Any,
        resume: bool = False,
        run_fingerprint: str | None = None,
        allow_replace: bool = False,
    ) -> None:
        self.final_path = Path(final_path)
        self.partial_path = Path(f"{self.final_path}.partial")
        self.lock_path = Path(f"{self.final_path}.partial.lock")
        self.expected_ordinals = _normalize_ordinals(expected_ordinals)
        self.run_fingerprint = (
            _DEFAULT_FINGERPRINT if run_fingerprint is None else run_fingerprint
        )
        if not isinstance(self.run_fingerprint, str):
            raise TypeError("run_fingerprint must be a string")
        if not self.run_fingerprint.strip():
            raise ValueError("run_fingerprint must not be empty")
        if not isinstance(resume, bool):
            raise TypeError("resume must be a boolean")
        if not isinstance(allow_replace, bool):
            raise TypeError("allow_replace must be a boolean")
        if resume and allow_replace:
            raise ValueError("resume and allow_replace are mutually exclusive")
        self.allow_replace = allow_replace

        self._completed: set[int] = set()
        self._next_index = 0
        self._handle = None
        self._lock_handle = None
        self._closed = False
        self._finalized = False

        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            self._initialize(resume)
        except BaseException:
            self._release_lock()
            raise

    @property
    def completed(self) -> frozenset[int]:
        return frozenset(self._completed)

    def _acquire_lock(self) -> None:
        self._lock_handle = self.lock_path.open("a+b")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock_handle.close()
            self._lock_handle = None
            if error.errno in (errno.EACCES, errno.EAGAIN):
                raise RuntimeError(f"checkpoint target is locked: {self.final_path}") from error
            raise

    def _release_lock(self) -> None:
        if self._lock_handle is None:
            return
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_handle.close()
            self._lock_handle = None

    def _initialize(self, resume: bool) -> None:
        final_exists = self.final_path.exists()
        partial_exists = self.partial_path.exists()
        if final_exists and partial_exists:
            raise FileExistsError(
                f"ambiguous checkpoint: both final and partial exist for {self.final_path}"
            )
        if not resume and (partial_exists or (final_exists and not self.allow_replace)):
            raise FileExistsError(f"checkpoint output already exists: {self.final_path}")

        if final_exists and resume:
            rows = self._read_committed_rows(self.final_path, truncate_torn_tail=False)
            self._validate_prefix(rows, require_complete=True)
            self._finalized = True
            self._closed = True
            self._release_lock()
            return

        if partial_exists:
            rows = self._read_committed_rows(self.partial_path, truncate_torn_tail=True)
            self._validate_prefix(rows, require_complete=False)
            self._handle = self.partial_path.open("ab")
            return

        self._handle = self.partial_path.open("xb")

    def _read_committed_rows(
        self, path: Path, *, truncate_torn_tail: bool
    ) -> list[dict[str, Any]]:
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            if not truncate_torn_tail:
                raise ValueError("complete checkpoint has an uncommitted tail")
            newline = data.rfind(b"\n")
            data = b"" if newline < 0 else data[: newline + 1]
            with path.open("r+b") as handle:
                handle.truncate(len(data))
                handle.flush()
                os.fsync(handle.fileno())
        return [_strict_json_object(line) for line in data.splitlines()]

    def _validate_prefix(
        self, rows: list[dict[str, Any]], *, require_complete: bool
    ) -> None:
        if len(rows) > len(self.expected_ordinals):
            raise ValueError("checkpoint contains out-of-range ordinals")
        for index, row in enumerate(rows):
            ordinal = row.get(_ORDINAL_KEY)
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise TypeError("checkpoint ordinal must be a non-boolean integer")
            if ordinal != self.expected_ordinals[index]:
                if ordinal in self._completed:
                    raise ValueError(f"duplicate ordinal: {ordinal}")
                raise ValueError("checkpoint ordinals are not the expected prefix")
            if row.get(_FINGERPRINT_KEY) != self.run_fingerprint:
                raise ValueError("checkpoint run fingerprint does not match")
            self._completed.add(ordinal)
        self._next_index = len(rows)
        if require_complete and self._next_index != len(self.expected_ordinals):
            raise ValueError("final checkpoint is incomplete")

    def write(self, ordinal: int, record: Mapping[str, Any]) -> None:
        if self._closed or self._handle is None:
            raise ValueError("checkpoint writer is closed")
        if isinstance(ordinal, bool) or not isinstance(ordinal, Integral):
            raise TypeError("ordinal must be a non-boolean integer")
        ordinal = int(ordinal)
        if ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if ordinal in self._completed:
            raise ValueError(f"duplicate ordinal: {ordinal}")
        if self._next_index >= len(self.expected_ordinals):
            raise ValueError(f"out-of-range ordinal: {ordinal}")
        if ordinal != self.expected_ordinals[self._next_index]:
            raise ValueError(f"out-of-order ordinal: {ordinal}")
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        collision = _RESERVED_KEYS.intersection(record)
        if collision:
            raise ValueError(f"record contains reserved keys: {sorted(collision)}")

        payload = dict(record)
        payload[_ORDINAL_KEY] = ordinal
        payload[_FINGERPRINT_KEY] = self.run_fingerprint
        encoded = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        _strict_json_object(encoded[:-1])
        self._handle.write(encoded)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._completed.add(ordinal)
        self._next_index += 1

    def finalize(self) -> None:
        if self._finalized:
            return
        if self._closed or self._handle is None:
            raise ValueError("checkpoint writer is closed")
        if self._next_index != len(self.expected_ordinals):
            raise ValueError("cannot finalize incomplete output")

        self._handle.close()
        self._handle = None
        replaced = False
        try:
            os.replace(self.partial_path, self.final_path)
            replaced = True
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(self.final_path.parent, flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            self._closed = True
            self._finalized = replaced
            self._release_lock()

    def close(self) -> None:
        if self._closed:
            return
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._closed = True
        self._release_lock()

    def __enter__(self) -> "JsonlCheckpointWriter":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


__all__ = ["JsonlCheckpointWriter"]
