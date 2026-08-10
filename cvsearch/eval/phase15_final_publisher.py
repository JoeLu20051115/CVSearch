"""Authenticate, score, and atomically publish Phase-15 LOGIV_V2 bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping

from cvsearch.eval.phase6_final_scorer import _rename_directory_noreplace
from cvsearch.evidence_gap.answers import official_letter
from cvsearch.evidence_gap.provenance import canonical_sha256


SUPPORTED_BENCHMARKS = ("vstar", "hr-bench_4k", "hr-bench_8k")
PHASE15_SELECTOR_ID = "logiv-v2-b8-backtrack-b7-hr-normalization"
TRUSTED_PHASE15_FREEZE_SHA256 = (
    "b95d312d06d2a1a669afdf6e0b3f01acf38bfbe5febf292ee8b0c628922ac1af"
)
TRUSTED_ANNOTATION_SHA256 = {
    "vstar": "b5a08bd78489c11170002a7c79519ac64c26a4776d5a963a0ea9018d4a9c9126",
    "hr-bench_4k": "675c10e56ac8aef086ebdb86957035679e9a8b4381e287b9fe6abcab2045218c",
    "hr-bench_8k": "52289e9eb9682b67496baa0b4478548b59a940b6a4eaf4d4228567786961e652",
}
STRICT_BASELINES = {"vstar": 171, "hr-bench_4k": 616, "hr-bench_8k": 618}
_B8_RECORD_FIELDS = frozenset({
    "source_ordinal", "input_identity_sha256", "extracted_pair_sha256", "p0",
    "v2_decision", "b5_record_sha256", "b5_decision", "decision",
})
_B7_RECORD_FIELDS = frozenset({
    "source_ordinal", "input_identity_sha256", "extracted_pair_sha256", "p0",
    "v2_decision", "projection", "decision",
})
_RECOVERY_DECISION_FIELDS = frozenset({"action", "status", "output"})
_V2_DECISION_FIELDS = frozenset({
    "action", "status", "output", "stability_gain",
})
_HR_NORMALIZATION_DECISION_FIELDS = frozenset({
    "action", "status", "output", "frequency", "canonical_answer",
})


@dataclass(frozen=True)
class FrozenFileSnapshot:
    path: Path
    data: bytes
    sha256: str
    stat_seal: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class PreparedPhase15Bundle:
    benchmark: str
    selected_jsonl_bytes: bytes
    manifest_bytes: bytes
    source_snapshot: FrozenFileSnapshot
    source_manifest_snapshot: FrozenFileSnapshot
    annotation_snapshot: FrozenFileSnapshot

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_bytes)

    @property
    def score(self) -> dict[str, Any]:
        return self.manifest["score"]


@dataclass(frozen=True)
class PreparedPhase15Suite:
    bundle_values: tuple[PreparedPhase15Bundle, ...]
    manifest_bytes: bytes
    freeze_snapshot: FrozenFileSnapshot

    @property
    def bundles(self) -> dict[str, PreparedPhase15Bundle]:
        return {bundle.benchmark: bundle for bundle in self.bundle_values}

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_bytes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_seal(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _read_snapshot(path: Path) -> FrozenFileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Phase-15 input is not one readable regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Phase-15 input must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_seal(before) != _stat_seal(after):
            raise RuntimeError("Phase-15 input changed while taking its snapshot")
    finally:
        os.close(descriptor)
    path_stat = os.stat(path, follow_symlinks=False)
    if _stat_seal(after) != _stat_seal(path_stat):
        raise RuntimeError("Phase-15 input path changed while taking its snapshot")
    data = b"".join(chunks)
    return FrozenFileSnapshot(
        path, data, hashlib.sha256(data).hexdigest(), _stat_seal(after),
    )


def _snapshot_is_current(snapshot: FrozenFileSnapshot) -> bool:
    try:
        current = os.stat(snapshot.path, follow_symlinks=False)
    except OSError:
        return False
    return _stat_seal(current) == snapshot.stat_seal


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _load_jsonl_bytes(data: bytes) -> list[dict[str, Any]]:
    rows = []
    for line in data.decode("utf-8").splitlines():
        value = json.loads(line)
        if type(value) is not dict:
            raise ValueError("Phase-15 JSONL rows must be exact objects")
        rows.append(value)
    return rows


def _manifest_path(benchmark: str, artifact: Path) -> Path:
    suffix = (
        ".backtrack-manifest.json"
        if benchmark == "vstar" else ".normalization-manifest.json"
    )
    return Path(f"{artifact}{suffix}")


def _validate_decision(benchmark: str, value: Any) -> None:
    if type(value) is not dict:
        raise ValueError("Phase-15 decision must be an exact object")
    action = value.get("action")
    if benchmark == "vstar":
        expected = (
            _RECOVERY_DECISION_FIELDS
            if action in {"P0", "VSTAR_BACKTRACK"} else _V2_DECISION_FIELDS
        )
        allowed = {"P0", "VSTAR_BACKTRACK", "EXPAND", "ZOOM"}
    else:
        expected = (
            _HR_NORMALIZATION_DECISION_FIELDS
            if action in {"P0", "HR_NORMALIZE"} else _V2_DECISION_FIELDS
        )
        allowed = {"P0", "HR_NORMALIZE", "EXPAND", "ZOOM"}
    if action not in allowed or set(value) != expected:
        raise ValueError("Phase-15 decision has an invalid exact action schema")
    if type(value.get("status")) is not str or value.get("output") is None:
        raise ValueError("Phase-15 decision status or output is invalid")


def _selected_bytes(benchmark: str, records: list[dict[str, Any]]) -> bytes:
    lines = []
    for record in records:
        decision = record["decision"]
        _validate_decision(benchmark, decision)
        lines.append(_canonical_bytes({
            "schema_version": 1,
            "_eg_ordinal": record["source_ordinal"],
            "output": decision["output"],
            "selection": {
                "selector_id": PHASE15_SELECTOR_ID,
                "action": decision["action"],
                "status": decision["status"],
                "source_record_sha256": canonical_sha256(record),
            },
        }))
    return b"".join(line + b"\n" for line in lines)


def _score(
    benchmark: str, records: list[dict[str, Any]], annotations: list[Any],
) -> dict[str, Any]:
    if len(records) != len(annotations):
        raise ValueError("Phase-15 predictions and annotations differ in length")
    correct = 0
    for ordinal, (record, annotation) in enumerate(zip(records, annotations)):
        if record.get("source_ordinal") != ordinal or type(annotation) is not dict:
            raise ValueError("Phase-15 prediction/annotation order is invalid")
        output = record["decision"]["output"]
        _validate_decision(benchmark, record["decision"])
        if benchmark == "vstar":
            options = annotation.get("options")
            if (
                type(output) is not int or type(options) is not list
                or not 0 <= output < len(options)
            ):
                raise ValueError("Phase-15 Vstar output is invalid")
            correct += output == 0
        else:
            truth = annotation.get("answer")
            if (
                type(output) is not list or len(output) != 4
                or not all(type(item) is str for item in output)
                or type(truth) is not list or len(truth) != 4
                or not all(item in "ABCD" for item in truth)
            ):
                raise ValueError("Phase-15 HR output or truth is invalid")
            correct += sum(
                want == official_letter(got) for want, got in zip(truth, output)
            )
    total = len(records) if benchmark == "vstar" else 4 * len(records)
    return {
        "correct": int(correct), "total": total,
        "accuracy_percent": 100.0 * correct / total,
        "strict_baseline_correct": STRICT_BASELINES[benchmark],
        "strictly_improved": correct > STRICT_BASELINES[benchmark],
    }


def _build_bundle_manifest(
    benchmark: str, *, freeze_snapshot: FrozenFileSnapshot,
    freeze: dict[str, Any], source_snapshot: FrozenFileSnapshot,
    source_manifest_snapshot: FrozenFileSnapshot,
    source_manifest: dict[str, Any], annotation_snapshot: FrozenFileSnapshot,
    selected_bytes: bytes, outputs: list[Any], score: dict[str, Any],
) -> dict[str, Any]:
    expected = freeze["benchmarks"][benchmark]
    return {
        "schema_version": 1,
        "artifact_kind": "phase15-logiv-v2-selected-bundle",
        "benchmark": benchmark,
        "selector_id": PHASE15_SELECTOR_ID,
        "freeze": {
            "path": str(freeze_snapshot.path),
            "sha256": freeze_snapshot.sha256,
            "joint_selected_output_sha256": freeze["joint_selected_output_sha256"],
        },
        "source_prediction": {
            "path": str(source_snapshot.path),
            "sha256": source_snapshot.sha256,
            "manifest_path": str(source_manifest_snapshot.path),
            "manifest_file_sha256": source_manifest_snapshot.sha256,
            "manifest_sha256": canonical_sha256(source_manifest),
            "selected_output_sha256": expected["selected_output_sha256"],
        },
        "trusted_annotation": {
            "path": str(annotation_snapshot.path),
            "sha256": annotation_snapshot.sha256,
            "size": len(annotation_snapshot.data),
        },
        "selected_jsonl": {
            "path": "selected.jsonl",
            "size": len(selected_bytes),
            "sha256": hashlib.sha256(selected_bytes).hexdigest(),
            "output_digest": canonical_sha256(outputs),
        },
        "score": score,
        "publisher_source_sha256": _sha256_file(Path(__file__)),
    }


def _build_suite_manifest(
    freeze_snapshot: FrozenFileSnapshot, freeze: dict[str, Any],
    bundles: Mapping[str, PreparedPhase15Bundle],
) -> dict[str, Any]:
    scores = {name: bundles[name].score for name in SUPPORTED_BENCHMARKS}
    return {
        "schema_version": 1,
        "artifact_kind": "phase15-logiv-v2-selected-suite",
        "selector_id": PHASE15_SELECTOR_ID,
        "freeze_sha256": freeze_snapshot.sha256,
        "joint_selected_output_sha256": freeze["joint_selected_output_sha256"],
        "benchmarks": scores,
        "bundle_manifest_sha256": {
            name: canonical_sha256(bundles[name].manifest)
            for name in SUPPORTED_BENCHMARKS
        },
        "gate": {
            "all_three_strictly_improved": all(
                score["strictly_improved"] for score in scores.values()
            ),
        },
    }


def prepare_phase15_suite(
    freeze_path: str | Path, *, artifacts: Mapping[str, Path],
    annotations: Mapping[str, Path],
) -> PreparedPhase15Suite:
    freeze_path = Path(freeze_path)
    freeze_snapshot = _read_snapshot(freeze_path)
    if freeze_snapshot.sha256 != TRUSTED_PHASE15_FREEZE_SHA256:
        raise ValueError("Phase-15 freeze is not the trusted committed report")
    freeze = json.loads(freeze_snapshot.data)
    if (
        type(freeze) is not dict
        or freeze.get("stage")
        != "phase15-b8-vstar-plus-b7-hr-complete-full-freeze"
        or freeze.get("joint_selected_output_sha256") is None
    ):
        raise ValueError("trusted Phase-15 freeze has invalid content")
    if set(artifacts) != set(SUPPORTED_BENCHMARKS) or set(annotations) != set(SUPPORTED_BENCHMARKS):
        raise ValueError("Phase-15 suite requires exactly three benchmarks")

    bundles: dict[str, PreparedPhase15Bundle] = {}
    joint_outputs = []
    for benchmark in SUPPORTED_BENCHMARKS:
        artifact = Path(artifacts[benchmark])
        source_manifest_path = _manifest_path(benchmark, artifact)
        annotation_path = Path(annotations[benchmark])
        expected = freeze["benchmarks"][benchmark]

        source_snapshot = _read_snapshot(artifact)
        source_manifest_snapshot = _read_snapshot(source_manifest_path)
        annotation_snapshot = _read_snapshot(annotation_path)
        artifact_sha256 = source_snapshot.sha256
        source_manifest_file_sha256 = source_manifest_snapshot.sha256
        if (
            artifact_sha256 != expected["output_file_sha256"]
            or source_manifest_file_sha256 != expected["manifest_file_sha256"]
        ):
            raise ValueError("Phase-15 prediction bytes differ from the trusted freeze")
        annotation_sha256 = annotation_snapshot.sha256
        if annotation_sha256 != TRUSTED_ANNOTATION_SHA256[benchmark]:
            raise ValueError("Phase-15 annotation bytes are not trusted")

        records = _load_jsonl_bytes(source_snapshot.data)
        source_manifests = _load_jsonl_bytes(source_manifest_snapshot.data)
        if len(source_manifests) != 1:
            raise ValueError("Phase-15 source manifest must contain one object")
        source_manifest = source_manifests[0]
        expected_fields = _B8_RECORD_FIELDS if benchmark == "vstar" else _B7_RECORD_FIELDS
        if any(set(record) != expected_fields for record in records):
            raise ValueError("Phase-15 prediction record has unknown or missing metadata")
        outputs = [record["decision"]["output"] for record in records]
        if (
            source_manifest.get("output_sha256") != artifact_sha256
            or source_manifest.get("selected_output_sha256")
            != expected["selected_output_sha256"]
            or canonical_sha256(outputs) != expected["selected_output_sha256"]
            or len(records) != expected["topics"]
        ):
            raise ValueError("Phase-15 prediction manifest or vector differs")
        annotations_value = json.loads(annotation_snapshot.data)
        if type(annotations_value) is not list:
            raise ValueError("Phase-15 annotation must be one JSON array")
        score = _score(benchmark, records, annotations_value)
        selected_bytes = _selected_bytes(benchmark, records)
        manifest = _build_bundle_manifest(
            benchmark,
            freeze_snapshot=freeze_snapshot,
            freeze=freeze,
            source_snapshot=source_snapshot,
            source_manifest_snapshot=source_manifest_snapshot,
            source_manifest=source_manifest,
            annotation_snapshot=annotation_snapshot,
            selected_bytes=selected_bytes,
            outputs=outputs,
            score=score,
        )
        bundles[benchmark] = PreparedPhase15Bundle(
            benchmark, selected_bytes, _canonical_bytes(manifest),
            source_snapshot, source_manifest_snapshot, annotation_snapshot,
        )
        joint_outputs.append(outputs)
    if canonical_sha256(joint_outputs) != freeze["joint_selected_output_sha256"]:
        raise ValueError("Phase-15 suite joint vector differs from the trusted freeze")
    suite_manifest = _build_suite_manifest(freeze_snapshot, freeze, bundles)
    if not suite_manifest["gate"]["all_three_strictly_improved"]:
        raise ValueError("Phase-15 suite does not strictly improve all three baselines")
    return PreparedPhase15Suite(
        tuple(bundles[name] for name in SUPPORTED_BENCHMARKS),
        _canonical_bytes(suite_manifest), freeze_snapshot,
    )


def _write_synced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_prepared_suite(
    prepared: PreparedPhase15Suite,
) -> dict[str, PreparedPhase15Bundle]:
    if not isinstance(prepared, PreparedPhase15Suite):
        raise TypeError("Phase-15 publication requires one prepared suite")
    bundles = prepared.bundles
    if set(bundles) != set(SUPPORTED_BENCHMARKS):
        raise ValueError("prepared Phase-15 suite has incomplete bundles")
    suite_manifest = prepared.manifest
    if _canonical_bytes(suite_manifest) != prepared.manifest_bytes:
        raise ValueError("prepared Phase-15 suite manifest is not canonical")
    freeze_data_sha256 = hashlib.sha256(
        prepared.freeze_snapshot.data,
    ).hexdigest()
    if (
        freeze_data_sha256 != TRUSTED_PHASE15_FREEZE_SHA256
        or prepared.freeze_snapshot.sha256 != freeze_data_sha256
        or not _snapshot_is_current(prepared.freeze_snapshot)
        or suite_manifest.get("freeze_sha256") != TRUSTED_PHASE15_FREEZE_SHA256
    ):
        raise RuntimeError("trusted Phase-15 freeze changed before publication")
    freeze = json.loads(prepared.freeze_snapshot.data)
    if (
        type(freeze) is not dict
        or freeze.get("stage")
        != "phase15-b8-vstar-plus-b7-hr-complete-full-freeze"
    ):
        raise ValueError("prepared Phase-15 freeze content is invalid")
    joint_outputs = []
    for benchmark in SUPPORTED_BENCHMARKS:
        bundle = bundles[benchmark]
        if not all(_snapshot_is_current(snapshot) for snapshot in (
            bundle.source_snapshot, bundle.source_manifest_snapshot,
            bundle.annotation_snapshot,
        )):
            raise RuntimeError("Phase-15 source changed before publication")
        expected = freeze["benchmarks"][benchmark]
        source_data_sha256 = hashlib.sha256(
            bundle.source_snapshot.data,
        ).hexdigest()
        source_manifest_data_sha256 = hashlib.sha256(
            bundle.source_manifest_snapshot.data,
        ).hexdigest()
        annotation_data_sha256 = hashlib.sha256(
            bundle.annotation_snapshot.data,
        ).hexdigest()
        if (
            bundle.source_snapshot.sha256 != source_data_sha256
            or source_data_sha256 != expected["output_file_sha256"]
            or bundle.source_manifest_snapshot.sha256
            != source_manifest_data_sha256
            or source_manifest_data_sha256 != expected["manifest_file_sha256"]
            or bundle.annotation_snapshot.sha256 != annotation_data_sha256
            or annotation_data_sha256 != TRUSTED_ANNOTATION_SHA256[benchmark]
        ):
            raise ValueError("prepared Phase-15 bytes are not trusted artifacts")
        manifest = bundle.manifest
        if _canonical_bytes(manifest) != bundle.manifest_bytes:
            raise ValueError("prepared Phase-15 bundle manifest is not canonical")
        records = _load_jsonl_bytes(bundle.source_snapshot.data)
        expected_fields = (
            _B8_RECORD_FIELDS if benchmark == "vstar" else _B7_RECORD_FIELDS
        )
        if any(set(record) != expected_fields for record in records):
            raise ValueError("prepared Phase-15 record has unknown or missing metadata")
        source_manifests = _load_jsonl_bytes(
            bundle.source_manifest_snapshot.data,
        )
        if len(source_manifests) != 1:
            raise ValueError("prepared Phase-15 source manifest is invalid")
        source_manifest = source_manifests[0]
        annotations = json.loads(bundle.annotation_snapshot.data)
        if type(annotations) is not list:
            raise ValueError("prepared Phase-15 annotation snapshot is invalid")
        score = _score(benchmark, records, annotations)
        selected_bytes = _selected_bytes(benchmark, records)
        outputs = [record["decision"]["output"] for record in records]
        expected_manifest = _build_bundle_manifest(
            benchmark,
            freeze_snapshot=prepared.freeze_snapshot,
            freeze=freeze,
            source_snapshot=bundle.source_snapshot,
            source_manifest_snapshot=bundle.source_manifest_snapshot,
            source_manifest=source_manifest,
            annotation_snapshot=bundle.annotation_snapshot,
            selected_bytes=selected_bytes,
            outputs=outputs,
            score=score,
        )
        if (
            bundle.selected_jsonl_bytes != selected_bytes
            or bundle.manifest_bytes != _canonical_bytes(expected_manifest)
            or manifest.get("score") != score
            or manifest.get("freeze", {}).get("sha256")
            != TRUSTED_PHASE15_FREEZE_SHA256
            or manifest.get("source_prediction", {}).get("sha256")
            != source_data_sha256
            or manifest.get("source_prediction", {}).get("manifest_file_sha256")
            != source_manifest_data_sha256
            or manifest.get("trusted_annotation", {}).get("sha256")
            != annotation_data_sha256
            or source_manifest.get("output_sha256") != source_data_sha256
            or source_manifest.get("selected_output_sha256")
            != expected["selected_output_sha256"]
            or canonical_sha256(outputs) != expected["selected_output_sha256"]
            or len(records) != expected["topics"]
            or manifest.get("selected_jsonl", {}).get("sha256")
            != hashlib.sha256(selected_bytes).hexdigest()
            or manifest.get("selected_jsonl", {}).get("size") != len(selected_bytes)
            or manifest.get("selected_jsonl", {}).get("output_digest")
            != canonical_sha256(outputs)
            or suite_manifest.get("benchmarks", {}).get(benchmark) != score
            or suite_manifest.get("bundle_manifest_sha256", {}).get(benchmark)
            != canonical_sha256(manifest)
            or not score["strictly_improved"]
        ):
            raise ValueError("prepared Phase-15 bundle differs from authenticated inputs")
        joint_outputs.append(outputs)
    expected_suite_manifest = _build_suite_manifest(
        prepared.freeze_snapshot, freeze, bundles,
    )
    if (
        canonical_sha256(joint_outputs)
        != freeze.get("joint_selected_output_sha256")
        or prepared.manifest_bytes != _canonical_bytes(expected_suite_manifest)
        or suite_manifest.get("joint_selected_output_sha256")
        != freeze.get("joint_selected_output_sha256")
        or suite_manifest.get("gate", {}).get("all_three_strictly_improved") is not True
    ):
        raise ValueError("Phase-15 strict-gain publication gate failed")
    return bundles


def publish_phase15_suite(
    output_root: str | Path, prepared: PreparedPhase15Suite,
) -> None:
    bundles = _validate_prepared_suite(prepared)
    output_root = Path(output_root)
    if output_root.name in {"", ".", ".."}:
        raise ValueError("Phase-15 output root is too broad")
    if output_root.is_symlink() or output_root.exists():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.tmp-", dir=output_root.parent,
    ))
    try:
        for benchmark, bundle in bundles.items():
            directory = temporary / benchmark
            directory.mkdir()
            _write_synced(directory / "selected.jsonl", bundle.selected_jsonl_bytes)
            _write_synced(
                directory / "selection-manifest.json",
                bundle.manifest_bytes + b"\n",
            )
        _write_synced(
            temporary / "suite-manifest.json",
            prepared.manifest_bytes + b"\n",
        )
        directory_fd = os.open(
            temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _validate_prepared_suite(prepared)
        _rename_directory_noreplace(temporary, output_root)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
