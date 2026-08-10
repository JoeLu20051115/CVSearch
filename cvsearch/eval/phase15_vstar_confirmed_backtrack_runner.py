#!/usr/bin/env python3
"""Replay frozen B5 Vstar evidence through confirmed-BACKTRACK admission."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cvsearch.eval.phase2_oracle import load_jsonl, load_launch_manifest
from cvsearch.eval.phase5_unified_selector import select_unified_state
from cvsearch.eval.phase6_combined_selection import (
    ExtractedCombinedPair,
    validate_and_extract_combined_pairs,
)
from cvsearch.eval.phase7_confirmation_runner import _atomic_write, _sha256_file
from cvsearch.eval.phase12_generated_query_lazy_runner import (
    LAZY_PROCESSOR_MODE,
    LOCALIZATION_QUERY_PROMPT_VERSION,
    LOCALIZATION_QUERY_TEMPLATE_SHA256,
    _candidate_dto,
    _decisions,
    is_eligible_lazy_pair,
)
from cvsearch.eval.phase12_generated_query_lazy_search import (
    all_lazy_rule_keys,
    project_lazy_candidate,
)
from cvsearch.eval.phase15_vstar_confirmed_backtrack import (
    B8_FROZEN_B5_RULE,
    select_vstar_confirmed_backtrack,
)
from cvsearch.evidence_gap.provenance import canonical_sha256


_TRUSTED_B5_FREEZE_SHA256 = frozenset({
    "1540da5560e019e25967faa35e1c62ce84c6971a35684ae75dacb2b60495901f",
    "3bba60ac48908d6cb17cdeffe933d3ca3e2666c3656a0150b7d93f395eb0666d",
})
_LEGACY_B5_MANIFEST_FIELDS = frozenset({
    "schema_version", "artifact_kind", "benchmark", "inference_revision",
    "base_jsonl", "combined_jsonl", "base_manifest_sha256",
    "combined_manifest_sha256", "qwen_artifact", "clip_artifact",
    "prompt_version", "prompt_template_sha256", "processor_mode", "rules",
    "records", "planned_calls", "runner_source_sha256",
    "selector_source_sha256", "output_sha256",
})
_MODERN_B5_MANIFEST_FIELDS = _LEGACY_B5_MANIFEST_FIELDS | frozenset({
    "eligible_records_total", "partition",
})
_B5_PARTITION_FIELDS = frozenset({
    "num_chunks", "chunk_idx", "ordinals", "ordinals_sha256",
})
_B5_RECORD_FIELDS = frozenset({
    "source_ordinal", "input_identity_sha256", "extracted_pair_sha256", "p0",
    "candidate", "decisions", "query_result", "query_sha256", "rank_material",
    "rank_sha256", "render_audits", "observations", "projection",
    "used_backtrack", "failure", "question_sha256", "options_sha256",
    "source_image", "cost",
})


def _load_trusted_b5_freeze(path: Path) -> dict[str, Any]:
    report_sha256 = _sha256_file(path)
    if report_sha256 not in _TRUSTED_B5_FREEZE_SHA256:
        raise ValueError("B5 freeze report is not a trusted precommitted artifact")
    report = json.loads(path.read_text())
    if type(report) is not dict:
        raise ValueError("trusted B5 freeze report must be one JSON object")
    stage = report.get("stage")
    benchmark = report.get("benchmarks", {}).get("vstar", {})
    artifacts = report.get("artifacts", {})
    if stage == "phase12-b5-generated-query-lazy-search":
        expected = [{
            "output_file_sha256": benchmark["output_sha256"],
            "manifest_file_sha256": benchmark["manifest_file_sha256"],
            "manifest_sha256": benchmark["manifest_sha256"],
        }]
        contract = {
            "qwen_model_sha256": artifacts["qwen_model_sha256"],
            "clip_model_sha256": artifacts["clip_model_sha256"],
            "prompt_template_sha256": artifacts["prompt_template_sha256"],
            "runner_source_sha256": artifacts["runner_source_sha256"],
            "selector_source_sha256": artifacts["selector_source_sha256"],
        }
    elif stage == "phase14-b7-complete-full-freeze":
        expected = [{
            "output_file_sha256": item["output_file_sha256"],
            "manifest_file_sha256": item["manifest_file_sha256"],
            "manifest_sha256": item["manifest_sha256"],
        } for item in benchmark["chunks"]]
        contract = {
            "qwen_model_sha256": artifacts["qwen_model_sha256"],
            "clip_model_sha256": artifacts["clip_model_sha256"],
            "prompt_template_sha256": artifacts["vstar_prompt_template_sha256"],
            "runner_source_sha256": artifacts["vstar_runner_source_sha256"],
            "selector_source_sha256": artifacts["vstar_selector_source_sha256"],
        }
    else:
        raise ValueError("trusted B5 freeze report has an unsupported stage")
    return {
        "path": str(path),
        "sha256": report_sha256,
        "stage": stage,
        "implementation_commit": report.get("implementation_commit"),
        "expected_artifacts": expected,
        "generation_contract": contract,
    }


def _validate_b5_manifest_schema(manifest: Any) -> str:
    if type(manifest) is not dict:
        raise ValueError("B5 manifest schema must be an exact object")
    fields = set(manifest)
    if fields == _LEGACY_B5_MANIFEST_FIELDS:
        kind = "legacy"
    elif fields == _MODERN_B5_MANIFEST_FIELDS:
        kind = "modern"
    else:
        raise ValueError("B5 manifest schema has missing or unknown fields")
    if manifest.get("schema_version") != 1:
        raise ValueError("B5 manifest schema version is unsupported")
    if kind == "modern":
        partition = manifest["partition"]
        if type(partition) is not dict or set(partition) != _B5_PARTITION_FIELDS:
            raise ValueError("modern B5 partition has an invalid exact schema")
    return kind


def _validate_generation_contract(
    manifest: Mapping[str, Any], trusted: Mapping[str, Any],
) -> None:
    contract = trusted["generation_contract"]
    if (
        manifest.get("qwen_artifact", {}).get("sha256")
        != contract["qwen_model_sha256"]
        or manifest.get("clip_artifact", {}).get("sha256")
        != contract["clip_model_sha256"]
        or manifest.get("prompt_version") != LOCALIZATION_QUERY_PROMPT_VERSION
        or manifest.get("prompt_template_sha256")
        != contract["prompt_template_sha256"]
        or manifest.get("prompt_template_sha256")
        != LOCALIZATION_QUERY_TEMPLATE_SHA256
        or manifest.get("processor_mode") != LAZY_PROCESSOR_MODE
        or manifest.get("rules") != list(all_lazy_rule_keys())
        or manifest.get("runner_source_sha256")
        != contract["runner_source_sha256"]
        or manifest.get("selector_source_sha256")
        != contract["selector_source_sha256"]
    ):
        raise ValueError("B5 generation contract differs from the trusted freeze")


def _validate_partition_layout(
    eligible_ordinals: Sequence[int],
    artifacts: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
) -> dict[int, Mapping[str, Any]]:
    if not artifacts:
        raise ValueError("B5 partition artifacts are missing")
    if len(artifacts) == 1 and "partition" not in artifacts[0][0]:
        records = artifacts[0][1]
        recorded = [record.get("source_ordinal") for record in records]
        if recorded != list(eligible_ordinals):
            raise ValueError("legacy B5 partition does not exactly cover eligibility")
        return dict(zip(recorded, records))
    if any(
        type(manifest.get("partition")) is not dict
        or set(manifest["partition"]) != _B5_PARTITION_FIELDS
        for manifest, _ in artifacts
    ):
        raise ValueError("modern B5 partition has an invalid exact schema")
    counts = {item[0].get("partition", {}).get("num_chunks") for item in artifacts}
    if len(counts) != 1:
        raise ValueError("B5 partition counts disagree")
    num_chunks = counts.pop()
    if type(num_chunks) is not int or num_chunks != len(artifacts) or num_chunks <= 0:
        raise ValueError("B5 partition set is incomplete")
    by_index: dict[int, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]] = {}
    for manifest, records in artifacts:
        partition = manifest.get("partition")
        if not isinstance(partition, Mapping):
            raise ValueError("B5 partition metadata is invalid")
        index = partition.get("chunk_idx")
        if type(index) is not int or not 0 <= index < num_chunks or index in by_index:
            raise ValueError("B5 partition index is invalid or duplicated")
        by_index[index] = (manifest, records)
    if set(by_index) != set(range(num_chunks)):
        raise ValueError("B5 partition indexes are incomplete")
    indexed: dict[int, Mapping[str, Any]] = {}
    eligible = list(eligible_ordinals)
    for index in range(num_chunks):
        manifest, records = by_index[index]
        expected = eligible[index::num_chunks]
        recorded = [record.get("source_ordinal") for record in records]
        partition = manifest["partition"]
        if partition.get("ordinals") != expected or recorded != expected:
            raise ValueError("B5 partition does not match its exact strided eligibility")
        if partition.get("ordinals_sha256") != canonical_sha256(expected):
            raise ValueError("B5 partition ordinal hash is invalid")
        for ordinal, record in zip(recorded, records):
            if ordinal in indexed:
                raise ValueError("B5 partition records overlap")
            indexed[ordinal] = record
    if set(indexed) != set(eligible):
        raise ValueError("B5 partition union does not cover eligibility")
    return indexed


def _validate_b5_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any], record: Mapping[str, Any],
) -> None:
    if type(record) is not dict or set(record) != _B5_RECORD_FIELDS:
        raise ValueError("B5 record has an invalid exact schema")
    if (
        record.get("source_ordinal") != pair.ordinal
        or record.get("input_identity_sha256") != canonical_sha256(pair.input_identity)
        or record.get("extracted_pair_sha256") != pair.extracted_digest
        or record.get("p0") != pair.p0
        or record.get("question_sha256")
        != hashlib.sha256(row["question"].encode()).hexdigest()
        or record.get("options_sha256") != canonical_sha256(row["options"])
        or record.get("rank_sha256") != canonical_sha256(record.get("rank_material"))
        or record.get("failure") is not None
    ):
        raise ValueError("B5 record does not bind one valid Vstar input pair")
    projection = project_lazy_candidate(
        row["answer_type"], row["options"], record.get("observations"),
    )
    candidate = _candidate_dto(
        projection,
        view_sha256=[item["sheet_sha256"] for item in record.get("render_audits", ())],
        rank_sha256=record["rank_sha256"], query_sha256=record["query_sha256"],
    )
    if (
        record.get("projection") != projection
        or record.get("candidate") != candidate
        or record.get("decisions") != _decisions(pair.p0, candidate)
        or B8_FROZEN_B5_RULE not in record["decisions"]
        or record.get("used_backtrack") != (len(record["observations"]) == 3)
    ):
        raise ValueError("B5 record projection or decision replay differs")


def _produce_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    b5_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    v2 = select_unified_state(pair.p0, pair.candidates)
    if v2.action == "P0" and b5_record is not None:
        frozen_b5 = b5_record["decisions"][B8_FROZEN_B5_RULE]
        decision = asdict(select_vstar_confirmed_backtrack(
            pair.p0, row["options"], b5_record["observations"], frozen_b5,
        ))
        b5_sha256 = canonical_sha256(b5_record)
    else:
        decision = asdict(v2)
        frozen_b5 = None
        b5_sha256 = None
    return {
        "source_ordinal": pair.ordinal,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "p0": pair.p0,
        "v2_decision": asdict(v2),
        "b5_record_sha256": b5_sha256,
        "b5_decision": frozen_b5,
        "decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
    parser.add_argument("--b5-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--b5-freeze-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_rows = load_jsonl(args.base_jsonl)
    combined_rows = load_jsonl(args.combined_jsonl)
    base_manifest = load_launch_manifest(args.base_jsonl)
    combined_manifest = load_launch_manifest(args.combined_jsonl)
    revision = base_manifest["code"]["revision"]
    pairs = validate_and_extract_combined_pairs(
        "vstar", base_rows, combined_rows,
        disabled_launch_manifest=base_manifest,
        enabled_launch_manifest=combined_manifest,
        expected_inference_revision=revision,
    )
    rows = {row["_eg_ordinal"]: row for row in base_rows}
    if len(rows) != len(base_rows) or set(rows) != {pair.ordinal for pair in pairs}:
        raise ValueError("B8 rows do not exactly cover validated v2 pairs")
    eligible = [
        pair for pair in pairs if is_eligible_lazy_pair(pair, rows[pair.ordinal])
    ]
    trusted_freeze = _load_trusted_b5_freeze(args.b5_freeze_report)
    expected_artifacts = {
        (
            item["output_file_sha256"], item["manifest_file_sha256"],
            item["manifest_sha256"],
        )
        for item in trusted_freeze["expected_artifacts"]
    }
    expected_raw_artifacts = {
        (item[0], item[1]) for item in expected_artifacts
    }
    authenticated_raw_artifacts = set()
    authenticated_artifacts = set()
    artifacts = []
    artifact_manifests = []
    for path in args.b5_jsonl:
        manifest_path = Path(f"{path}.lazy-manifest.json")
        output_file_sha256 = _sha256_file(path)
        manifest_file_sha256 = _sha256_file(manifest_path)
        authenticated_raw = (output_file_sha256, manifest_file_sha256)
        if (
            authenticated_raw not in expected_raw_artifacts
            or authenticated_raw in authenticated_raw_artifacts
        ):
            raise ValueError(
                "B5 raw bytes are not authenticated by the trusted freeze"
            )
        authenticated_raw_artifacts.add(authenticated_raw)
        records = load_jsonl(path)
        manifest = load_jsonl(manifest_path)[0]
        schema_kind = _validate_b5_manifest_schema(manifest)
        manifest_sha256 = canonical_sha256(manifest)
        authenticated = (
            output_file_sha256, manifest_file_sha256, manifest_sha256,
        )
        if (
            authenticated not in expected_artifacts
            or authenticated in authenticated_artifacts
        ):
            raise ValueError("B5 artifact is not authenticated by the trusted freeze")
        authenticated_artifacts.add(authenticated)
        _validate_generation_contract(manifest, trusted_freeze)
        if (
            manifest.get("artifact_kind") != "phase12-generated-query-lazy-search"
            or manifest.get("benchmark") != "vstar"
            or manifest.get("inference_revision") != revision
            or manifest.get("output_sha256") != output_file_sha256
            or manifest.get("records") != len(records)
            or (
                schema_kind == "modern"
                and manifest.get("eligible_records_total") != len(eligible)
            )
            or manifest.get("base_jsonl", {}).get("sha256") != _sha256_file(args.base_jsonl)
            or manifest.get("combined_jsonl", {}).get("sha256")
            != _sha256_file(args.combined_jsonl)
            or manifest.get("base_manifest_sha256") != canonical_sha256(base_manifest)
            or manifest.get("combined_manifest_sha256")
            != canonical_sha256(combined_manifest)
            or B8_FROZEN_B5_RULE not in manifest.get("rules", ())
        ):
            raise ValueError("B5 artifact or manifest differs from B8 inputs")
        artifacts.append((manifest, records))
        artifact_manifests.append({
            "path": str(path),
            "sha256": output_file_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_sha256": manifest_sha256,
            "partition": manifest.get("partition") or {
                "legacy_singleton": True,
                "num_chunks": 1,
                "chunk_idx": 0,
                "ordinals": [record["source_ordinal"] for record in records],
                "ordinals_sha256": canonical_sha256([
                    record["source_ordinal"] for record in records
                ]),
            },
        })
    if authenticated_artifacts != expected_artifacts:
        raise ValueError("B5 artifacts do not exactly cover the trusted freeze")
    b5_by_ordinal = _validate_partition_layout(
        [pair.ordinal for pair in eligible], artifacts,
    )
    eligible_by_ordinal = {pair.ordinal: pair for pair in eligible}
    for ordinal, record in b5_by_ordinal.items():
        _validate_b5_record(
            eligible_by_ordinal[ordinal], rows[ordinal], record,
        )
    records = [
        _produce_record(pair, rows[pair.ordinal], b5_by_ordinal.get(pair.ordinal))
        for pair in pairs
    ]
    output_sha256 = _atomic_write(args.output, records)
    v2_outputs = [record["v2_decision"]["output"] for record in records]
    selected_outputs = [record["decision"]["output"] for record in records]
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase15-vstar-confirmed-backtrack",
        "benchmark": "vstar",
        "inference_revision": revision,
        "base_jsonl": {"path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl)},
        "combined_jsonl": {
            "path": str(args.combined_jsonl), "sha256": _sha256_file(args.combined_jsonl),
        },
        "base_manifest_file_sha256": _sha256_file(Path(
            f"{args.base_jsonl}.launch-manifest.json"
        )),
        "base_manifest_sha256": canonical_sha256(base_manifest),
        "combined_manifest_file_sha256": _sha256_file(Path(
            f"{args.combined_jsonl}.launch-manifest.json"
        )),
        "combined_manifest_sha256": canonical_sha256(combined_manifest),
        "trusted_b5_freeze": trusted_freeze,
        "b5_artifacts": artifact_manifests,
        "selection_rule": {
            "frozen_b5_rule": B8_FROZEN_B5_RULE,
            "trajectory": ["P0", "candidate", "candidate"],
            "requires_split_backtrack_disagreement_recovery": True,
            "option_permutation_invariant": True,
        },
        "records": len(records),
        "eligible_b5_records": len(eligible),
        "selected_confirmed_backtracks": sum(
            record["decision"]["action"] == "VSTAR_BACKTRACK"
            for record in records
        ),
        "v2_output_sha256": canonical_sha256(v2_outputs),
        "selected_output_sha256": canonical_sha256(selected_outputs),
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase15_vstar_confirmed_backtrack.py")
        ),
        "output_sha256": output_sha256,
    }
    manifest_path = Path(f"{args.output}.backtrack-manifest.json")
    _atomic_write(manifest_path, [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
