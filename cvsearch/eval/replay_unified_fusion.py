"""Read-only HR development replay for global soft evidence fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.answers import official_letter
from cvsearch.evidence_gap.fusion import soft_fuse_hr
from cvsearch.evidence_gap.types import AnswerRecord


GAMMAS = (0.0, 1.1, 1.5, 2.1, 4.1)
_LETTERS = frozenset("ABCD")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _snapshot(path: Path) -> tuple[int, str]:
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read JSONL input: {path}") from error
    if not lines:
        raise ValueError("JSONL input must not be empty")
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        try:
            row = json.loads(
                line,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
            json.dumps(row, allow_nan=False)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid JSONL row at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        rows.append(row)
    return rows


def _answer_record(payload: Any, context: str) -> AnswerRecord:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} answer must be an object")
    copied = deepcopy(dict(payload))
    copied["raw_outputs"] = tuple(copied.get("raw_outputs", ()))
    copied["losses"] = tuple(copied.get("losses", ()))
    try:
        return AnswerRecord(**copied)
    except TypeError as error:
        raise ValueError(f"{context} answer is not an AnswerRecord payload") from error


def _history(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trace = row.get("method_trace")
    if not isinstance(trace, Mapping):
        raise ValueError("row has no method_trace")
    history = trace.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("method_trace.history must be a nonempty list")
    if any(not isinstance(item, Mapping) or "answer" not in item for item in history):
        raise ValueError("method_trace.history must contain answer records")
    return history


def _selected_source(row: Mapping[str, Any]) -> str:
    trace = row["method_trace"]
    assert isinstance(trace, Mapping)
    final = trace.get("final_answer")
    if not isinstance(final, Mapping):
        raise ValueError("method_trace.final_answer must be an object")
    selected = final.get("selected_from")
    if selected not in {"root", "search"}:
        raise ValueError("final selected source must be root or search")
    return selected


def select_evidence(row: Mapping[str, Any]) -> AnswerRecord:
    """Return the answer record that was selected at the time of evaluation."""
    history = _history(row)
    selected = _selected_source(row)
    matches = [item["answer"] for item in history if item["answer"].get("selected_from") == selected]
    if not matches:
        raise ValueError(f"method_trace has no {selected} answer record")
    return _answer_record(matches[-1], "selected history")


def _raw_anchor(row: Mapping[str, Any]) -> tuple[str, ...]:
    history = _history(row)
    for source in ("search", "root"):
        matches = [item["answer"] for item in history if item["answer"].get("selected_from") == source]
        if matches:
            raw_outputs = matches[-1].get("raw_outputs")
            if (
                not isinstance(raw_outputs, list)
                or len(raw_outputs) != 4
                or any(not isinstance(output, str) for output in raw_outputs)
            ):
                raise ValueError(f"latest {source} history answer must contain four raw outputs")
            return tuple(raw_outputs)
    raise ValueError("method_trace has neither search nor root answer history")


def _validate_row(row: Mapping[str, Any], seen_ordinals: set[int]) -> None:
    ordinal = row.get("_eg_ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise ValueError("_eg_ordinal must be an integer")
    if ordinal in seen_ordinals:
        raise ValueError(f"duplicate _eg_ordinal: {ordinal}")
    seen_ordinals.add(ordinal)
    for field in ("answer", "options", "output"):
        value = row.get(field)
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError(f"row {ordinal} must contain four {field}")
    if any(answer not in _LETTERS for answer in row["answer"]):
        raise ValueError(f"row {ordinal} answers must be A-D")
    if any(not isinstance(option, str) for option in row["options"]):
        raise ValueError(f"row {ordinal} options must be strings")
    if any(not isinstance(output, str) for output in row["output"]):
        raise ValueError(f"row {ordinal} outputs must be strings")
    _history(row)
    _selected_source(row)
    _raw_anchor(row)
    select_evidence(row)


def topic_accuracy(row: Mapping[str, Any], output: Sequence[str]) -> float:
    if len(output) != 4:
        raise ValueError("replayed output must contain four cycles")
    return sum(answer == official_letter(choice) for answer, choice in zip(row["answer"], output)) / 4.0


def _config_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    encoded = set()
    for row in rows:
        trace = row["method_trace"]
        assert isinstance(trace, Mapping)
        config = trace.get("effective_config")
        if not isinstance(config, Mapping):
            raise ValueError("method_trace.effective_config must be an object")
        try:
            encoded.add(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("effective_config must be strict JSON") from error
    if len(encoded) != 1:
        raise ValueError("all rows for a resolution must share one effective_config")
    return hashlib.sha256(encoded.pop().encode("utf-8")).hexdigest()


def _gamma_values(gammas: Sequence[float]) -> tuple[float, ...]:
    if not gammas:
        raise ValueError("at least one gamma is required")
    result = []
    for gamma in gammas:
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
            raise TypeError("gamma must be a finite non-negative number")
        value = float(gamma)
        if not math.isfinite(value) or value < 0:
            raise ValueError("gamma must be a finite non-negative number")
        if value in result:
            raise ValueError("gammas must be unique")
        result.append(value)
    return tuple(result)


def _gamma_key(gamma: float) -> str:
    return str(gamma)


def _replay_resolution(path: Path, gammas: tuple[float, ...]) -> dict[str, Any]:
    rows = _load_jsonl(path)
    seen_ordinals: set[int] = set()
    for row in rows:
        _validate_row(row, seen_ordinals)
    anchors = [_raw_anchor(row) for row in rows]
    evidence = [select_evidence(row) for row in rows]
    anchor_correct_cycles = sum(
        sum(answer == official_letter(choice) for answer, choice in zip(row["answer"], anchor))
        for row, anchor in zip(rows, anchors)
    )
    anchor_accuracy = anchor_correct_cycles / (len(rows) * 4)
    candidates: dict[str, dict[str, Any]] = {}
    for gamma in gammas:
        outputs = [
            tuple(soft_fuse_hr(row["options"], anchor, record, gamma).output)
            for row, anchor, record in zip(rows, anchors, evidence)
        ]
        correct_cycles = sum(
            sum(answer == official_letter(choice) for answer, choice in zip(row["answer"], output))
            for row, output in zip(rows, outputs)
        )
        accuracy = correct_cycles / (len(rows) * 4)
        candidates[_gamma_key(gamma)] = {
            "correct_cycles": correct_cycles,
            "accuracy": accuracy,
            "anchor_correct_cycles": anchor_correct_cycles,
            "anchor_accuracy": anchor_accuracy,
            "delta": accuracy - anchor_accuracy,
            "changed_topics": sum(output != anchor for output, anchor in zip(outputs, anchors)),
        }
    return {
        "config_hash": _config_hash(rows),
        "n_topics": len(rows),
        "n_cycles": len(rows) * 4,
        "anchor_correct_cycles": anchor_correct_cycles,
        "anchor_accuracy": anchor_accuracy,
        "candidates": candidates,
    }


def replay_paths(paths: Mapping[str, str | Path], gammas: Sequence[float] = GAMMAS) -> dict[str, Any]:
    """Replay one global fusion weight per HR topic without changing JSONL inputs."""
    if not paths:
        raise ValueError("at least one HR JSONL path is required")
    if any(not isinstance(name, str) or not name.startswith("hr-bench_") for name in paths):
        raise ValueError("paths must be keyed by HR benchmark name")
    gamma_values = _gamma_values(gammas)
    source_paths = {name: Path(path) for name, path in paths.items()}
    before = {name: _snapshot(path) for name, path in source_paths.items()}
    try:
        reports = {name: _replay_resolution(path, gamma_values) for name, path in source_paths.items()}
    finally:
        changed = [name for name, path in source_paths.items() if _snapshot(path) != before[name]]
        if changed:
            raise RuntimeError(f"input JSONL changed during replay: {', '.join(changed)}")

    global_candidates: dict[str, dict[str, Any]] = {}
    for gamma in gamma_values:
        key = _gamma_key(gamma)
        deltas = [report["candidates"][key]["delta"] for report in reports.values()]
        global_candidates[key] = {
            "min_delta": min(deltas),
            "changed_topic_count": sum(report["candidates"][key]["changed_topics"] for report in reports.values()),
        }
    selected_gamma = max(
        gamma_values,
        key=lambda gamma: (global_candidates[_gamma_key(gamma)]["min_delta"], -gamma),
    )
    selected = global_candidates[_gamma_key(selected_gamma)]
    config_material = {
        name: report["config_hash"] for name, report in sorted(reports.items())
    }
    report: dict[str, Any] = {
        "calibration_status": "exploratory_development",
        "config_hash": hashlib.sha256(
            json.dumps(config_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "candidates": global_candidates,
        "min_delta": selected["min_delta"],
        "changed_topic_count": selected["changed_topic_count"],
        "selected_gamma": selected_gamma,
    }
    report.update(reports)
    return report


def _parse_gammas(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gammas must be comma-separated numbers") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr4", required=True, type=Path)
    parser.add_argument("--hr8", required=True, type=Path)
    parser.add_argument("--gammas", type=_parse_gammas, default=GAMMAS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.hr4.resolve(), args.hr8.resolve()}:
        parser.error("--output must not overwrite an input JSONL")
    report = replay_paths({"hr-bench_4k": args.hr4, "hr-bench_8k": args.hr8}, args.gammas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
