"""Leak-free scoring and atomic materialization for frozen Phase-6 decisions."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import fcntl

import cvsearch.eval.phase3_zoom_oracle as phase3
import cvsearch.eval.phase6_combined_selection as phase6
from cvsearch.eval.phase2_oracle import (
    FROZEN_DEV_ORDINAL_SHA256,
    _hr_correct,
)
from cvsearch.eval.phase6_combined_selection import (
    FrozenCombinedDecisionBatch,
    freeze_combined_decisions,
)
from cvsearch.evidence_gap.provenance import canonical_sha256


ROOT = Path(__file__).resolve().parents[2]
_SUPPORTED_BENCHMARKS = ("vstar", "hr-bench_4k", "hr-bench_8k")
_HR_BENCHMARKS = frozenset({"hr-bench_4k", "hr-bench_8k"})
_SHA256_CHARS = frozenset("0123456789abcdef")


def _ordinal_sha256(ordinals: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in ordinals).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str or len(value) != 64
        or any(char not in _SHA256_CHARS for char in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class Phase6Expectation:
    benchmark: str
    profile: str
    split: str
    topics: int
    cycles: int
    ordinal_sha256: str
    locked_cvsearch_correct: int

    def __post_init__(self) -> None:
        if self.benchmark not in _SUPPORTED_BENCHMARKS:
            raise ValueError("Phase-6 expectation benchmark is unsupported")
        if self.profile not in {"dev", "full"}:
            raise ValueError("Phase-6 expectation profile must be dev or full")
        if self.split != ("dev" if self.profile == "dev" else "all"):
            raise ValueError("Phase-6 expectation split/profile disagree")
        for name in ("topics", "cycles", "locked_cvsearch_correct"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Phase-6 expectation {name} is invalid")
        per_topic = 4 if self.benchmark in _HR_BENCHMARKS else 1
        if (
            self.topics < 1 or self.cycles != self.topics * per_topic
            or self.locked_cvsearch_correct > self.cycles
        ):
            raise ValueError("Phase-6 expectation counts are inconsistent")
        _sha256(self.ordinal_sha256, "Phase-6 expectation ordinal digest")


PROFILE_EXPECTATIONS: dict[tuple[str, str], Phase6Expectation] = {
    ("dev", "vstar"): Phase6Expectation(
        "vstar", "dev", "dev", 37, 37,
        FROZEN_DEV_ORDINAL_SHA256["vstar"], 32,
    ),
    ("dev", "hr-bench_4k"): Phase6Expectation(
        "hr-bench_4k", "dev", "dev", 39, 156,
        FROZEN_DEV_ORDINAL_SHA256["hr-bench_4k"], 106,
    ),
    ("dev", "hr-bench_8k"): Phase6Expectation(
        "hr-bench_8k", "dev", "dev", 28, 112,
        FROZEN_DEV_ORDINAL_SHA256["hr-bench_8k"], 90,
    ),
    ("full", "vstar"): Phase6Expectation(
        "vstar", "full", "all", 191, 191,
        _ordinal_sha256(tuple(range(191))), 170,
    ),
    ("full", "hr-bench_4k"): Phase6Expectation(
        "hr-bench_4k", "full", "all", 200, 800,
        _ordinal_sha256(tuple(range(200))), 613,
    ),
    ("full", "hr-bench_8k"): Phase6Expectation(
        "hr-bench_8k", "full", "all", 200, 800,
        _ordinal_sha256(tuple(range(200))), 614,
    ),
}


@dataclass(frozen=True)
class PreparedSelectedPartition:
    expectation: Phase6Expectation
    frozen_batch: FrozenCombinedDecisionBatch
    raw_identity: Mapping[str, Any]
    annotation_claim: Mapping[str, Any]
    label_blind_input_sha256: str
    selected_rows: tuple[Mapping[str, Any], ...]
    selected_jsonl_bytes: bytes
    selected_jsonl_sha256: str
    selected_output_digest: str
    evaluator_revision: Mapping[str, Any]
    canonical_digest: str


@dataclass(frozen=True)
class ScoredSelectedPartition:
    prepared: PreparedSelectedPartition
    report: Mapping[str, Any]
    derived_manifest: Mapping[str, Any]
    trusted_annotation_stat: tuple[int, int, int, int, int]


_SOURCE_PATHS = tuple(
    ROOT / relative for relative in (
        "cvsearch/eval/phase2_oracle.py",
        "cvsearch/eval/phase3_zoom_oracle.py",
        "cvsearch/eval/phase4_expand_oracle.py",
        "cvsearch/eval/phase5_unified_selector.py",
        "cvsearch/eval/phase6_combined_selection.py",
        "cvsearch/eval/phase6_final_scorer.py",
        "cvsearch/evidence_gap/answers.py",
        "cvsearch/evidence_gap/provenance.py",
    )
)
_SOURCE_HASHES_AT_IMPORT = {
    path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in _SOURCE_PATHS
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _snapshot_json(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _snapshot_file(path: Path) -> tuple[dict[str, Any], bytes]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"required Phase-6 source is not an exact regular file: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"required Phase-6 source is not an exact regular file: {path}"
            )
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    snapshot = lambda value: (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )
    if snapshot(before) != snapshot(after):
        raise RuntimeError(f"Phase-6 source changed during snapshot: {path}")
    try:
        current = path.lstat()
    except OSError as error:
        raise RuntimeError(f"Phase-6 source changed during snapshot: {path}") from error
    if stat.S_ISLNK(current.st_mode) or snapshot(current) != snapshot(before):
        raise RuntimeError(f"Phase-6 source changed during snapshot: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError(f"Phase-6 source changed during snapshot: {path}")
    identity = {
        "path": os.path.abspath(os.fspath(path)), "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return identity, raw


def _file_identity(path: Path) -> dict[str, Any]:
    return _snapshot_file(path)[0]


def _annotation_stat_seal(path: Path) -> tuple[int, int, int, int, int]:
    try:
        value = path.lstat()
    except OSError as error:
        raise RuntimeError("trusted Phase-6 annotation is no longer available") from error
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError("trusted Phase-6 annotation is no longer a regular file")
    return (
        value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _parse_jsonl_snapshot(raw: bytes, source: Path) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"JSONL must be nonempty and newline-terminated: {source}")
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL line {line_number}: {source}")
        try:
            row = json.loads(
                line.decode("utf-8"), parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
            _canonical_bytes(row)
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid JSONL line {line_number}: {source}") from error
        if type(row) is not dict:
            raise TypeError(f"JSONL line {line_number} must be an exact object")
        rows.append(row)
    return rows


def _parse_manifest_snapshot(raw: bytes, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        _canonical_bytes(value)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid launch manifest: {source}") from error
    if type(value) is not dict:
        raise TypeError(f"launch manifest must be an exact object: {source}")
    return value


def _evaluator_revision() -> dict[str, Any]:
    files = [{
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in _SOURCE_PATHS]
    current = {item["path"]: item["sha256"] for item in files}
    if current != _SOURCE_HASHES_AT_IMPORT:
        raise RuntimeError("Phase-6 scorer dependency changed during evaluation")
    return {
        "kind": "source_manifest_sha256", "files": files,
        "sha256": canonical_sha256(files),
    }


def _label_blind_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        if type(row) is not dict:
            raise TypeError("Phase-6 raw rows must be exact JSON objects")
        projected.append(_snapshot_json({
            key: row[key] for key in row if key != "answer"
        }))
    return projected


def _validate_partition(
    expectation: Phase6Expectation, manifest: Mapping[str, Any],
    ordinals: Sequence[int],
) -> None:
    partition = phase3._mapping(
        manifest.get("selected_partition"), "Phase-6 selected partition",
    )
    expected_ordinals = list(ordinals)
    if (
        len(expected_ordinals) != expectation.topics
        or partition.get("ordinals") != expected_ordinals
        or partition.get("rows") != expectation.topics
        or _ordinal_sha256(expected_ordinals) != expectation.ordinal_sha256
        or partition.get("split") != expectation.split
        or partition.get("split_seed") != 260809
        or partition.get("num_chunks") != 1
        or partition.get("chunk_idx") != 0
    ):
        raise ValueError("Phase-6 raw partition differs from the formal expectation")


def _annotation_claim(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifact = phase3._mapping(
        phase3._mapping(manifest.get("artifacts"), "Phase-6 artifacts").get(
            "annotation_file"
        ),
        "Phase-6 annotation artifact",
    )
    files = phase3._list(artifact.get("files"), "Phase-6 annotation files")
    if len(files) != 1:
        raise ValueError("Phase-6 annotation artifact must contain one file")
    entry = phase3._mapping(files[0], "Phase-6 annotation file")
    path = artifact.get("path")
    size = entry.get("size")
    if (
        type(path) is not str or not path
        or isinstance(size, bool) or not isinstance(size, int) or size < 0
    ):
        raise ValueError("Phase-6 annotation claim is invalid")
    return {
        "path": path, "size": size,
        "file_sha256": _sha256(entry.get("sha256"), "annotation file digest"),
        "artifact_sha256": _sha256(
            artifact.get("sha256"), "annotation artifact digest",
        ),
        "selected_rows_sha256": _sha256(
            phase3._mapping(
                manifest.get("selected_partition"), "Phase-6 partition",
            ).get("rows_sha256"),
            "annotation selected-row digest",
        ),
    }


def _validate_output(benchmark: str, options: Any, output: Any) -> Any:
    if benchmark == "vstar":
        if type(output) is not int or not isinstance(options, list) or not 0 <= output < len(options):
            raise ValueError("Phase-6 V* selected output is outside exact options")
        return output
    if (
        type(output) is not list or len(output) != 4
        or any(type(item) is not str for item in output)
    ):
        raise ValueError("Phase-6 HR selected output must contain four exact strings")
    return output


def _admitted_by_frozen_rule(
    action: str, gain: float, batch: FrozenCombinedDecisionBatch,
) -> bool:
    matches = [rule for rule in batch.admission_rules if rule.action == action]
    if len(matches) != 1:
        raise ValueError("Phase-6 selected action has no unique admission rule")
    rule = matches[0]
    if rule.operator == ">=":
        return gain >= rule.threshold
    if rule.operator == ">":
        return gain > rule.threshold
    raise ValueError("Phase-6 admission rule operator is unsupported")


def _selected_material(
    expectation: Phase6Expectation, batch: FrozenCombinedDecisionBatch,
) -> tuple[tuple[dict[str, Any], ...], bytes, str, str, str]:
    batch.verify_digest()
    if (
        batch.selector_id != phase6.SELECTOR_ID
        or type(batch.admission_rules) is not tuple
        or batch.admission_rules != phase6.REVIEWED_ADMISSION_RULES
        or batch.tie_order != ("EXPAND", "ZOOM")
        or batch.selector_source_sha256 != phase6.SELECTOR_SOURCE_SHA256
    ):
        raise ValueError("Phase-6 frozen selector identity is not reviewed")
    rows = []
    inputs = []
    for record in batch:
        decision = record.decision
        input_identity = record.input_identity
        output = _snapshot_json(decision.output)
        _validate_output(expectation.benchmark, input_identity.get("options"), output)
        if decision.action == "P0":
            if decision.status != "retained_p0" or decision.stability_gain is not None:
                raise ValueError("Phase-6 retained P0 decision is not canonical")
        elif decision.action in {"ZOOM", "EXPAND"}:
            gain = decision.stability_gain
            if (
                decision.status != "selected_candidate"
                or isinstance(gain, bool) or not isinstance(gain, (int, float))
                or not math.isfinite(float(gain))
                or not _admitted_by_frozen_rule(
                    decision.action, float(gain), batch,
                )
            ):
                raise ValueError("Phase-6 candidate decision is not canonical")
        else:
            raise ValueError("Phase-6 selected action is unsupported")
        rows.append({
            "schema_version": 1,
            "_eg_ordinal": record.ordinal,
            "output": output,
            "selection": {
                "selector_id": batch.selector_id,
                "action": decision.action,
                "status": decision.status,
                "stability_gain": decision.stability_gain,
            },
        })
        inputs.append(input_identity)
    encoded = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    output_digest = canonical_sha256([
        {"ordinal": row["_eg_ordinal"], "output": row["output"]} for row in rows
    ])
    return (
        tuple(rows), encoded, hashlib.sha256(encoded).hexdigest(), output_digest,
        canonical_sha256(inputs),
    )


def _prepared_material(prepared: PreparedSelectedPartition) -> dict[str, Any]:
    return {
        "expectation": asdict(prepared.expectation),
        "frozen_batch_sha256": prepared.frozen_batch.canonical_digest,
        "raw_identity": prepared.raw_identity,
        "annotation_claim": prepared.annotation_claim,
        "label_blind_input_sha256": prepared.label_blind_input_sha256,
        "selected_rows": list(prepared.selected_rows),
        "selected_jsonl_sha256": prepared.selected_jsonl_sha256,
        "selected_output_digest": prepared.selected_output_digest,
        "evaluator_revision": prepared.evaluator_revision,
    }


def _verify_raw_snapshot(raw_identity: Mapping[str, Any]) -> None:
    for variant in ("disabled", "combined"):
        for kind in ("jsonl", "launch_manifest"):
            expected = phase3._mapping(
                phase3._mapping(raw_identity.get(variant), variant).get(kind), kind,
            )
            current = _file_identity(Path(expected["path"]))
            if current != expected:
                raise RuntimeError("Phase-6 raw input changed after decision freeze")


def _read_snapshotted_jsonl(expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = Path(expected["path"])
    identity, raw = _snapshot_file(source)
    if identity != expected:
        raise RuntimeError("Phase-6 raw JSONL changed after decision freeze")
    return _parse_jsonl_snapshot(raw, source)


def _read_snapshotted_manifest(expected: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(expected["path"])
    identity, raw = _snapshot_file(source)
    if identity != expected:
        raise RuntimeError("Phase-6 launch manifest changed after decision freeze")
    return _parse_manifest_snapshot(raw, source)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_trusted_annotation(claim: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(claim["path"])
    identity, raw = _snapshot_file(path)  # Deliberate unique trusted-label boundary.
    if (
        identity["size"] != claim["size"]
        or identity["sha256"] != claim["file_sha256"]
    ):
        raise RuntimeError("trusted Phase-6 annotation changed after decision freeze")
    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        _canonical_bytes(value)
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("trusted Phase-6 annotation is not strict UTF-8 JSON") from error
    if type(value) is not list or any(type(row) is not dict for row in value):
        raise TypeError("trusted Phase-6 annotation must contain exact objects")
    return value


def _score_output(
    benchmark: str, label: Any, output: Any, options: Any,
) -> tuple[int, Any]:
    _validate_output(benchmark, options, output)
    if benchmark == "vstar":
        if label is not None:
            raise ValueError("V* uses the trusted first-option label rule")
        return int(output == 0), output
    if (
        type(label) is not list or len(label) != 4
        or any(
            type(item) is not str or len(item) != 1 or item not in "ABCD"
            for item in label
        )
    ):
        raise ValueError("trusted HR labels must contain four exact A-D letters")
    correct, parsed = _hr_correct(label, output)
    return correct, parsed


def _cycle_transition_counts(
    benchmark: str, label: Any, p0_correct: int, selected_correct: int,
    p0_parsed: Any, selected_parsed: Any,
) -> tuple[int, int]:
    if benchmark == "vstar":
        return (
            int(p0_correct == 0 and selected_correct == 1),
            int(p0_correct == 1 and selected_correct == 0),
        )
    p0_cycles = tuple(want == got for want, got in zip(label, p0_parsed))
    selected_cycles = tuple(
        want == got for want, got in zip(label, selected_parsed)
    )
    if (
        len(p0_cycles) != 4 or len(selected_cycles) != 4
        or sum(p0_cycles) != p0_correct
        or sum(selected_cycles) != selected_correct
    ):
        raise RuntimeError("official HR parsed cycles differ from aggregate score")
    corrected = sum(
        not before and after
        for before, after in zip(p0_cycles, selected_cycles)
    )
    corrupted = sum(
        before and not after
        for before, after in zip(p0_cycles, selected_cycles)
    )
    return corrected, corrupted


def _fairness_gate(
    profile: str, paired_p0_correct: int, selected_correct: int,
    locked_cvsearch_correct: int,
) -> bool:
    return (
        profile == "full"
        and selected_correct > paired_p0_correct
        and selected_correct > locked_cvsearch_correct
    )


def _verify_prepared(prepared: PreparedSelectedPartition) -> None:
    if not isinstance(prepared, PreparedSelectedPartition):
        raise TypeError("Phase-6 prepared partition has an invalid type")
    prepared.frozen_batch.verify_digest()
    rows, encoded, selected_sha, output_digest, input_digest = _selected_material(
        prepared.expectation, prepared.frozen_batch,
    )
    if (
        rows != prepared.selected_rows
        or encoded != prepared.selected_jsonl_bytes
        or selected_sha != prepared.selected_jsonl_sha256
        or output_digest != prepared.selected_output_digest
        or input_digest != prepared.label_blind_input_sha256
        or hashlib.sha256(prepared.selected_jsonl_bytes).hexdigest()
        != prepared.selected_jsonl_sha256
        or canonical_sha256(_prepared_material(prepared)) != prepared.canonical_digest
    ):
        raise ValueError("Phase-6 prepared partition seal is invalid")
    _verify_raw_snapshot(prepared.raw_identity)
    if _evaluator_revision() != prepared.evaluator_revision:
        raise RuntimeError("Phase-6 evaluator changed after decision freeze")


def _source_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    files = [{
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in paths]
    return {
        "kind": "source_manifest_sha256", "files": files,
        "sha256": canonical_sha256(files),
    }


def _execution_identity(
    prepared: PreparedSelectedPartition, manifest: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = prepared.frozen_batch[0].provenance
    artifacts = phase3._mapping(manifest.get("artifacts"), "Phase-6 artifacts")
    artifact_hashes = {
        name: _sha256(
            phase3._mapping(artifacts.get(name), f"Phase-6 {name}").get("sha256"),
            f"Phase-6 {name} digest",
        )
        for name in ("qwen", "processor", "sam", "spacy")
    }
    config = phase3._mapping(manifest.get("config"), "Phase-6 config")
    disabled_manifest = _read_snapshotted_manifest(
        prepared.raw_identity["disabled"]["launch_manifest"],
    )
    disabled_config = phase3._mapping(
        disabled_manifest.get("config"), "Phase-6 disabled config",
    )
    environment = dict(phase3._mapping(
        manifest.get("environment"), "Phase-6 environment",
    ))
    runtime_environment = dict(environment)
    runtime_environment.pop("cuda_visible_devices", None)
    return {
        "inference_revision": provenance.revision,
        "paired_gpu_uuid": provenance.gpu_uuid,
        "model_contract": phase3._launch_model_contract(manifest),
        "model_runtime_artifact_sha256": artifact_hashes,
        "environment_sha256": canonical_sha256(environment),
        "runtime_environment_sha256": canonical_sha256(runtime_environment),
        "disabled_config_sha256": _sha256(
            disabled_config.get("loaded_sha256"), "disabled config digest",
        ),
        "combined_config_sha256": _sha256(
            config.get("loaded_sha256"), "combined config digest",
        ),
    }


def prepare_selected_partition_from_paths(
    benchmark: str, disabled_path: str | Path, combined_path: str | Path, *,
    profile: str,
) -> PreparedSelectedPartition:
    if (profile, benchmark) not in PROFILE_EXPECTATIONS:
        raise ValueError("Phase-6 scorer requires an exact dev/full expectation")
    expectation = PROFILE_EXPECTATIONS[(profile, benchmark)]
    disabled_path = Path(disabled_path)
    combined_path = Path(combined_path)
    disabled_sidecar = Path(f"{disabled_path}.launch-manifest.json")
    combined_sidecar = Path(f"{combined_path}.launch-manifest.json")
    evaluator = _evaluator_revision()
    disabled_jsonl_identity, disabled_raw = _snapshot_file(disabled_path)
    combined_jsonl_identity, combined_raw = _snapshot_file(combined_path)
    disabled_sidecar_identity, disabled_manifest_raw = _snapshot_file(
        disabled_sidecar,
    )
    combined_sidecar_identity, combined_manifest_raw = _snapshot_file(
        combined_sidecar,
    )
    disabled_rows = _parse_jsonl_snapshot(disabled_raw, disabled_path)
    combined_rows = _parse_jsonl_snapshot(combined_raw, combined_path)
    disabled_manifest = _parse_manifest_snapshot(
        disabled_manifest_raw, disabled_sidecar,
    )
    combined_manifest = _parse_manifest_snapshot(
        combined_manifest_raw, combined_sidecar,
    )
    raw_identity = {
        "disabled": {
            "jsonl": disabled_jsonl_identity,
            "launch_manifest": disabled_sidecar_identity,
            "parsed_manifest_fingerprint": canonical_sha256(disabled_manifest),
        },
        "combined": {
            "jsonl": combined_jsonl_identity,
            "launch_manifest": combined_sidecar_identity,
            "parsed_manifest_fingerprint": canonical_sha256(combined_manifest),
        },
    }
    blind_disabled = _label_blind_rows(disabled_rows)
    blind_combined = _label_blind_rows(combined_rows)
    batch = freeze_combined_decisions(
        benchmark, blind_disabled, blind_combined,
        disabled_launch_manifest=disabled_manifest,
        enabled_launch_manifest=combined_manifest,
    )
    _validate_partition(
        expectation, disabled_manifest,
        tuple(record.ordinal for record in batch),
    )
    rows, encoded, selected_sha, output_digest, input_digest = _selected_material(
        expectation, batch,
    )
    _verify_raw_snapshot(raw_identity)
    if _evaluator_revision() != evaluator:
        raise RuntimeError("Phase-6 evaluator changed during decision freeze")
    prepared = PreparedSelectedPartition(
        expectation=expectation,
        frozen_batch=batch,
        raw_identity=_snapshot_json(raw_identity),
        annotation_claim=_annotation_claim(combined_manifest),
        label_blind_input_sha256=input_digest,
        selected_rows=rows,
        selected_jsonl_bytes=encoded,
        selected_jsonl_sha256=selected_sha,
        selected_output_digest=output_digest,
        evaluator_revision=evaluator,
        canonical_digest="",
    )
    return PreparedSelectedPartition(
        **{
            **prepared.__dict__,
            "canonical_digest": canonical_sha256(_prepared_material(prepared)),
        }
    )


def _bind_prepared(
    prepared: PreparedSelectedPartition,
) -> ScoredSelectedPartition:
    _verify_prepared(prepared)
    annotation_path = Path(prepared.annotation_claim["path"])
    annotation_before = _annotation_stat_seal(annotation_path)
    annotations = _read_trusted_annotation(prepared.annotation_claim)
    # Embedded raw labels are deliberately reloaded only after the trusted boundary.
    disabled_rows = _read_snapshotted_jsonl(
        prepared.raw_identity["disabled"]["jsonl"],
    )
    combined_rows = _read_snapshotted_jsonl(
        prepared.raw_identity["combined"]["jsonl"],
    )
    _verify_raw_snapshot(prepared.raw_identity)
    selected_annotation_rows = []
    scored_rows = []
    for index, record in enumerate(prepared.frozen_batch):
        ordinal = record.ordinal
        if ordinal >= len(annotations):
            raise ValueError("Phase-6 ordinal is outside trusted annotation")
        annotation = annotations[ordinal]
        selected_annotation_rows.append({
            "ordinal": ordinal, "annotation": _snapshot_json(annotation),
        })
        input_identity = record.input_identity
        for field in ("input_image", "question", "options", "answer_type"):
            if input_identity.get(field) != annotation.get(field):
                raise ValueError(
                    f"Phase-6 frozen input differs from trusted annotation {field}"
                )
        disabled = disabled_rows[index]
        combined_row = combined_rows[index]
        if disabled.get("_eg_ordinal") != ordinal or combined_row.get("_eg_ordinal") != ordinal:
            raise ValueError("Phase-6 raw labels differ from frozen ordinal order")
        if prepared.expectation.benchmark == "vstar":
            if (
                "answer" in annotation or "answer" in disabled
                or "answer" in combined_row
            ):
                raise ValueError("V* must not expose an answer field")
            label = None
        else:
            label = annotation.get("answer")
            if (
                type(label) is not list or len(label) != 4
                or any(
                    type(item) is not str
                    or len(item) != 1 or item not in "ABCD"
                    for item in label
                )
                or disabled.get("answer") != label
                or combined_row.get("answer") != label
            ):
                raise ValueError("HR labels do not match the trusted four-cycle answer")
        p0_output = record.p0["output"]
        selected_output = record.decision.output
        options = input_identity["options"]
        p0_correct, p0_parsed = _score_output(
            prepared.expectation.benchmark, label, p0_output, options,
        )
        selected_correct, selected_parsed = _score_output(
            prepared.expectation.benchmark, label, selected_output, options,
        )
        corrected_cycles, corrupted_cycles = _cycle_transition_counts(
            prepared.expectation.benchmark, label,
            p0_correct, selected_correct, p0_parsed, selected_parsed,
        )
        state_scores = [{"action": "P0", "correct": p0_correct}]
        for candidate in record.candidates:
            if candidate["feasible"]:
                candidate_correct, _ = _score_output(
                    prepared.expectation.benchmark, label,
                    candidate["output"], options,
                )
                state_scores.append({
                    "action": candidate["action"], "correct": candidate_correct,
                })
        scored_rows.append({
            "ordinal": ordinal,
            "action": record.decision.action,
            "p0_correct": p0_correct,
            "selected_correct": selected_correct,
            "oracle_correct": max(item["correct"] for item in state_scores),
            "state_scores": state_scores,
            "changed": selected_parsed != p0_parsed,
            "corrected_cycles": corrected_cycles,
            "corrupted_cycles": corrupted_cycles,
            "total": 4 if prepared.expectation.benchmark in _HR_BENCHMARKS else 1,
        })
    selected_rows_sha = canonical_sha256(selected_annotation_rows)
    if selected_rows_sha != prepared.annotation_claim["selected_rows_sha256"]:
        raise ValueError("trusted annotation rows differ from launch partition")
    cycles = sum(row["total"] for row in scored_rows)
    if cycles != prepared.expectation.cycles:
        raise ValueError("Phase-6 scoring cycles differ from formal expectation")
    p0_correct = sum(row["p0_correct"] for row in scored_rows)
    selected_correct = sum(row["selected_correct"] for row in scored_rows)
    oracle_correct = sum(row["oracle_correct"] for row in scored_rows)
    corrected_cycles = sum(row["corrected_cycles"] for row in scored_rows)
    corrupted_cycles = sum(row["corrupted_cycles"] for row in scored_rows)
    action_counts = {
        action: sum(row["action"] == action for row in scored_rows)
        for action in ("P0", "ZOOM", "EXPAND")
    }
    exceeds_p0 = selected_correct > p0_correct
    exceeds_locked = selected_correct > prepared.expectation.locked_cvsearch_correct
    report = {
        "benchmark": prepared.expectation.benchmark,
        "profile": prepared.expectation.profile,
        "sample": {
            "topics": len(scored_rows), "cycles": cycles,
            "topic_unit": "input row; HR four shuffles remain bound",
        },
        "revision": prepared.frozen_batch[0].provenance.revision,
        "p0": {
            "correct": p0_correct, "total": cycles,
            "accuracy": p0_correct / cycles,
        },
        "selected": {
            "correct": selected_correct, "total": cycles,
            "accuracy": selected_correct / cycles,
            "delta_correct": selected_correct - p0_correct,
            "delta_accuracy": (selected_correct - p0_correct) / cycles,
            "action_counts": action_counts,
            "changed_topics": sum(row["changed"] for row in scored_rows),
            "corrected_topics": sum(
                row["selected_correct"] > row["p0_correct"] for row in scored_rows
            ),
            "corrupted_topics": sum(
                row["selected_correct"] < row["p0_correct"] for row in scored_rows
            ),
            "corrected_cycles": corrected_cycles,
            "corrupted_cycles": corrupted_cycles,
        },
        "oracle": {
            "correct": oracle_correct, "total": cycles,
            "accuracy": oracle_correct / cycles,
            "candidate_generation_headroom_only": True,
            "not_used_for_selection": True,
            "state_unit": "one complete state per topic; no HR cycle mixing",
        },
        "gate": {
            "locked_cvsearch_correct": prepared.expectation.locked_cvsearch_correct,
            "strictly_exceeds_paired_p0": exceeds_p0,
            "strictly_exceeds_locked_cvsearch": exceeds_locked,
            "formal_full_success": (
                _fairness_gate(
                    prepared.expectation.profile, p0_correct, selected_correct,
                    prepared.expectation.locked_cvsearch_correct,
                )
            ),
        },
    }
    combined_manifest = _read_snapshotted_manifest(
        prepared.raw_identity["combined"]["launch_manifest"],
    )
    execution = _execution_identity(prepared, combined_manifest)
    annotation_identity = {
        **_snapshot_json(prepared.annotation_claim),
        "selected_rows_sha256": selected_rows_sha,
        "loaded_after_complete_partition_freeze": True,
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase6-combined-selected-bundle",
        "benchmark": prepared.expectation.benchmark,
        "profile": prepared.expectation.profile,
        "partition": {
            "split": prepared.expectation.split,
            "split_seed": 260809, "num_chunks": 1, "chunk_idx": 0,
            "ordinals": [record.ordinal for record in prepared.frozen_batch],
            "topics": prepared.expectation.topics,
            "cycles": prepared.expectation.cycles,
            "annotation_rows_sha256": selected_rows_sha,
        },
        "raw_sources": _snapshot_json(prepared.raw_identity),
        "execution_identity": execution,
        "decision_freeze": {
            "selector_id": prepared.frozen_batch.selector_id,
            "selector_source_sha256": prepared.frozen_batch.selector_source_sha256,
            "admission_rules": [
                asdict(rule) for rule in prepared.frozen_batch.admission_rules
            ],
            "tie_order": list(prepared.frozen_batch.tie_order),
            "canonical_digest": prepared.frozen_batch.canonical_digest,
            "label_blind_input_sha256": prepared.label_blind_input_sha256,
            "completed_before_trusted_label_read": True,
            "validator_source_manifest": _source_manifest(tuple(
                ROOT / relative for relative in (
                    "cvsearch/eval/phase3_zoom_oracle.py",
                    "cvsearch/eval/phase4_expand_oracle.py",
                    "cvsearch/eval/phase5_unified_selector.py",
                    "cvsearch/eval/phase6_combined_selection.py",
                )
            )),
        },
        "trusted_annotation": annotation_identity,
        "selected_jsonl": {
            "path": "selected.jsonl",
            "size": len(prepared.selected_jsonl_bytes),
            "sha256": prepared.selected_jsonl_sha256,
            "output_digest": prepared.selected_output_digest,
        },
        "score": _snapshot_json(report),
        "scorer_source_manifest": _snapshot_json(prepared.evaluator_revision),
    }
    _verify_prepared(prepared)
    annotation_after = _annotation_stat_seal(annotation_path)
    if annotation_before != annotation_after:
        raise RuntimeError("trusted annotation changed during score construction")
    return ScoredSelectedPartition(
        prepared=prepared,
        report=_snapshot_json(report),
        derived_manifest=_snapshot_json(manifest),
        trusted_annotation_stat=annotation_after,
    )


def bind_and_score_trusted_annotation(
    benchmark: str, disabled_path: str | Path, combined_path: str | Path, *,
    profile: str,
) -> ScoredSelectedPartition:
    """Authoritatively freeze raw paths and cross the label boundary in one call."""
    prepared = prepare_selected_partition_from_paths(
        benchmark, disabled_path, combined_path, profile=profile,
    )
    return _bind_prepared(prepared)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise RuntimeError(
            "Phase-6 atomic publication requires renameat2(RENAME_NOREPLACE)"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(target), 1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), os.fspath(target),
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(target))


def _write_selected_bundle_atomic(
    bundle_path: str | Path, scored: ScoredSelectedPartition,
) -> None:
    if not isinstance(scored, ScoredSelectedPartition):
        raise TypeError("Phase-6 selected bundle requires a scored partition")
    _verify_prepared(scored.prepared)
    annotation_path = Path(scored.prepared.annotation_claim["path"])
    if _annotation_stat_seal(annotation_path) != scored.trusted_annotation_stat:
        raise RuntimeError("trusted annotation changed before bundle publication")
    if (
        scored.derived_manifest.get("score") != scored.report
        or phase3._mapping(
            scored.derived_manifest.get("selected_jsonl"),
            "Phase-6 selected manifest",
        ).get("sha256") != scored.prepared.selected_jsonl_sha256
    ):
        raise ValueError("Phase-6 scored manifest differs from frozen outputs")
    bundle_path = Path(bundle_path)
    if bundle_path.name in {"", ".", ".."}:
        raise ValueError("Phase-6 bundle target is too broad")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.is_symlink() or bundle_path.exists():
        raise FileExistsError(bundle_path)
    lock_path = bundle_path.with_name(f".{bundle_path.name}.lock")
    lock = lock_path.open("a+b")
    temporary: Path | None = None
    published = False
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Phase-6 bundle target is locked: {bundle_path}") from error
        if bundle_path.is_symlink() or bundle_path.exists():
            raise FileExistsError(bundle_path)
        temporary = Path(tempfile.mkdtemp(
            prefix=f".{bundle_path.name}.tmp-", dir=bundle_path.parent,
        ))
        manifest_bytes = _canonical_bytes(scored.derived_manifest) + b"\n"
        for name, payload in (
            ("selected.jsonl", scored.prepared.selected_jsonl_bytes),
            ("selection-manifest.json", manifest_bytes),
        ):
            path = temporary / name
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        directory_fd = os.open(
            temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if bundle_path.is_symlink() or bundle_path.exists():
            raise FileExistsError(bundle_path)
        if _annotation_stat_seal(annotation_path) != scored.trusted_annotation_stat:
            raise RuntimeError("trusted annotation changed before bundle publication")
        _rename_directory_noreplace(temporary, bundle_path)
        published = True
        temporary = None
        parent_fd = os.open(
            bundle_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()
    if not published:
        raise RuntimeError("Phase-6 selected bundle was not published")


def write_selected_bundle_atomic(
    bundle_path: str | Path, benchmark: str, disabled_path: str | Path,
    combined_path: str | Path, *, profile: str,
) -> ScoredSelectedPartition:
    """Authoritatively score raw paths and publish their selected bundle."""
    scored = bind_and_score_trusted_annotation(
        benchmark, disabled_path, combined_path, profile=profile,
    )
    _write_selected_bundle_atomic(bundle_path, scored)
    return scored


def _build_suite_report(
    scored: Mapping[str, ScoredSelectedPartition], profile: str,
) -> dict[str, Any]:
    if set(scored) != set(_SUPPORTED_BENCHMARKS):
        raise ValueError("Phase-6 suite requires exactly V*, HR-4K, and HR-8K")
    values = []
    for benchmark in _SUPPORTED_BENCHMARKS:
        value = scored[benchmark]
        if not isinstance(value, ScoredSelectedPartition):
            raise TypeError("Phase-6 suite entries must be scored partitions")
        if (
            value.report.get("benchmark") != benchmark
            or value.report.get("profile") != profile
            or value.derived_manifest.get("benchmark") != benchmark
            or value.derived_manifest.get("profile") != profile
            or value.derived_manifest.get("score") != value.report
        ):
            raise ValueError("Phase-6 suite benchmark/profile binding differs")
        values.append(value)
    revisions = {value.report["revision"] for value in values}
    shared_execution = {
        canonical_sha256({
            key: value.derived_manifest["execution_identity"][key]
            for key in (
                "model_contract", "model_runtime_artifact_sha256",
                "runtime_environment_sha256", "disabled_config_sha256",
                "combined_config_sha256",
            )
        })
        for value in values
    }
    shared_selector = {
        canonical_sha256({
            key: value.derived_manifest["decision_freeze"][key]
            for key in (
                "selector_id", "selector_source_sha256", "admission_rules",
                "tie_order", "validator_source_manifest",
            )
        })
        for value in values
    }
    scorer_sources = {
        canonical_sha256(value.derived_manifest["scorer_source_manifest"])
        for value in values
    }
    if (
        len(revisions) != 1 or len(shared_execution) != 1
        or len(shared_selector) != 1 or len(scorer_sources) != 1
    ):
        raise ValueError("Phase-6 suite does not share one frozen execution identity")
    per_dataset = {
        benchmark: bool(scored[benchmark].report["gate"]["formal_full_success"])
        for benchmark in _SUPPORTED_BENCHMARKS
    }
    return {
        "schema_version": 1,
        "task": "phase6-combined-frozen-selection-suite",
        "profile": profile,
        "execution_identity": {
            "inference_revision": next(iter(revisions)),
            "shared_model_runtime_sha256": next(iter(shared_execution)),
            "shared_selector_sha256": next(iter(shared_selector)),
            "shared_scorer_sha256": next(iter(scorer_sources)),
            "different_equivalent_gpu_per_dataset_allowed": True,
            "same_gpu_required_within_each_pair": True,
        },
        "benchmarks": {
            benchmark: _snapshot_json(scored[benchmark].report)
            for benchmark in _SUPPORTED_BENCHMARKS
        },
        "bundle_manifest_sha256": {
            benchmark: canonical_sha256(scored[benchmark].derived_manifest)
            for benchmark in _SUPPORTED_BENCHMARKS
        },
        "gate": {
            "per_dataset_full_success": per_dataset,
            "vstar_advantage_preserved": per_dataset["vstar"],
            "all_three_full_success": (
                profile == "full" and all(per_dataset.values())
            ),
        },
    }


def score_phase6_suite(
    paths: Mapping[str, tuple[str | Path, str | Path]],
    output_root: str | Path, *, profile: str,
) -> Mapping[str, Any]:
    if set(paths) != set(_SUPPORTED_BENCHMARKS):
        raise ValueError("Phase-6 suite paths must contain exactly three benchmarks")
    scored = {}
    for benchmark in _SUPPORTED_BENCHMARKS:
        pair = paths[benchmark]
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError("Phase-6 suite path pair must be an exact two-tuple")
        scored[benchmark] = bind_and_score_trusted_annotation(
            benchmark, pair[0], pair[1], profile=profile,
        )
    report = _build_suite_report(scored, profile)
    output_root = Path(output_root)
    for benchmark in _SUPPORTED_BENCHMARKS:
        _write_selected_bundle_atomic(output_root / benchmark, scored[benchmark])
    return report
