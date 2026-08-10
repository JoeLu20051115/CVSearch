#!/usr/bin/env python3
"""Re-answer frozen B5 HR evidence with schema-normalized semantic option loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cvsearch.eval.phase4_expand_oracle as phase4
from cvsearch.eval.phase2_oracle import load_jsonl, load_launch_manifest
from cvsearch.eval.phase6_combined_selection import (
    ExtractedCombinedPair,
    validate_and_extract_combined_pairs,
)
from cvsearch.eval.phase7_confirmation_runner import (
    MAX_CONFIRMATION_CALLS,
    _atomic_write,
    _load_model,
    _sha256_file,
)
from cvsearch.eval.phase12_generated_query_lazy_search import AdaptivePatch
from cvsearch.eval.phase12_generated_query_lazy_runner import (
    _render_patch,
    is_eligible_lazy_pair,
)
from cvsearch.eval.phase13_hr_semantic_option_loss import (
    HR_LOSS_MIN_CONFIDENCES,
    HR_LOSS_MIN_PATH_CONSENSUS,
    build_semantic_option_set,
    hr_loss_rule_key,
    project_hr_loss_candidate,
    select_hr_loss_candidate,
)
from cvsearch.evidence_gap.provenance import canonical_sha256, content_manifest


B5_FIXED_VSTAR_RULE = "p0.6666666666666666-c0.05-g-0.1"
HR_LOSS_PROCESSOR_MODE = "b5_parent_child_conditional_backtrack_semantic_option_loss"


def _answer_hr_loss(
    model: Any, row: Mapping[str, Any], sheet: Any,
    option_set: Mapping[str, Any],
) -> dict[str, Any]:
    winner, losses = model.multiple_choices_with_losses(
        sheet, row["question"], option_set["choices"], [],
    )
    return {"winner": int(winner), "losses": [float(value) for value in losses]}


def _observe_hr_loss_path(
    model: Any, row: Mapping[str, Any], sheets: Sequence[Any],
    option_set: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(sheets, (list, tuple)) or len(sheets) != 3:
        raise ValueError("HR option-loss path requires three available sheets")
    observations = [
        _answer_hr_loss(model, row, sheets[0], option_set),
        _answer_hr_loss(model, row, sheets[1], option_set),
    ]
    if observations[0]["winner"] == observations[1]["winner"]:
        return observations, False
    observations.append(_answer_hr_loss(model, row, sheets[2], option_set))
    return observations, True


def _candidate_dto(
    projection: Mapping[str, Any] | None, *, view_sha256: Sequence[str],
    b5_record_sha256: str, option_set_sha256: str,
) -> dict[str, Any]:
    feasible = bool(projection is not None and projection.get("feasible") is True)
    return {
        "action": "HR_LOSS",
        "feasible": feasible,
        "output": projection["output"] if feasible else None,
        "stability": {"confidence": projection["confidence"]} if feasible else None,
        "path_consensus": projection["path_consensus"] if feasible else None,
        "view_sha256": list(view_sha256),
        "b5_record_sha256": b5_record_sha256,
        "option_set_sha256": option_set_sha256,
    }


def _decisions(p0: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        hr_loss_rule_key(path, confidence): asdict(select_hr_loss_candidate(
            p0, candidate, min_path_consensus=path,
            min_confidence=confidence,
        ))
        for path in sorted(HR_LOSS_MIN_PATH_CONSENSUS)
        for confidence in sorted(HR_LOSS_MIN_CONFIDENCES)
    }


def _patch_from_rank(
    rank_material: Mapping[str, Any], collection: str, identity_key: str,
) -> AdaptivePatch:
    identity = rank_material[identity_key]
    matches = [
        item["patch"] for item in rank_material[collection]
        if item["patch"]["identity"] == identity
    ]
    if len(matches) != 1:
        raise ValueError("B5 rank material does not identify one frozen patch")
    patch = matches[0]
    return AdaptivePatch(tuple(patch["path"]), tuple(patch["box"]))


def _validate_b5_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
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
        raise ValueError("B5 record does not bind one valid HR input pair")
    candidate = record.get("candidate")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("rank_sha256") != record.get("rank_sha256")
        or candidate.get("query_sha256") != record.get("query_sha256")
    ):
        raise ValueError("B5 candidate does not bind its query/rank material")


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _produce_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    base_manifest: Mapping[str, Any], b5_record: Mapping[str, Any], model: Any,
) -> dict[str, Any]:
    _validate_b5_record(pair, row, b5_record)
    b5_record_sha256 = canonical_sha256(b5_record)
    option_set = build_semantic_option_set(row["options"])
    option_set_sha256 = canonical_sha256(option_set)
    render_audits = observations = projection = None
    used_backtrack = False
    failure = None
    candidate = None
    charged_calls = 0
    try:
        source, source_audit = phase4._resolve_source_image(row, base_manifest)
        if source_audit != b5_record["source_image"]:
            raise ValueError("source image changed after B5")
        rank = b5_record["rank_material"]
        patches = (
            _patch_from_rank(rank, "root_siblings", "focus_identity"),
            _patch_from_rank(rank, "split_siblings", "child_identity"),
            _patch_from_rank(rank, "root_siblings", "backtrack_identity"),
        )
        rendered = [_render_patch(source, patch) for patch in patches]
        sheets = [value[0] for value in rendered]
        all_audits = [value[1] for value in rendered]
        frozen_audits = b5_record["render_audits"]
        if all_audits[:len(frozen_audits)] != frozen_audits:
            raise ValueError("B5 rendered observations changed before HR option loss")
        observations, used_backtrack = _observe_hr_loss_path(
            model, row, sheets, option_set,
        )
        charged_calls = len(observations)
        render_audits = all_audits[:charged_calls]
        projection = project_hr_loss_candidate(option_set, observations)
        candidate = _candidate_dto(
            projection,
            view_sha256=[audit["sheet_sha256"] for audit in render_audits],
            b5_record_sha256=b5_record_sha256,
            option_set_sha256=option_set_sha256,
        )
    except Exception as error:
        failure = _failure(error)
        charged_calls = 3
    if candidate is None:
        candidate = _candidate_dto(
            None, view_sha256=[], b5_record_sha256=b5_record_sha256,
            option_set_sha256=option_set_sha256,
        )
    return {
        "source_ordinal": pair.ordinal,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "b5_record_sha256": b5_record_sha256,
        "p0": pair.p0,
        "option_set": option_set,
        "option_set_sha256": option_set_sha256,
        "candidate": candidate,
        "decisions": _decisions(pair.p0, candidate),
        "render_audits": render_audits,
        "observations": observations,
        "projection": projection,
        "used_backtrack": used_backtrack,
        "failure": failure,
        "cost": {"planned_calls": 3, "charged_calls": charged_calls},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("hr-bench_4k", "hr-bench_8k"), required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
    parser.add_argument("--b5-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_rows = load_jsonl(args.base_jsonl)
    combined_rows = load_jsonl(args.combined_jsonl)
    base_manifest = load_launch_manifest(args.base_jsonl)
    combined_manifest = load_launch_manifest(args.combined_jsonl)
    b5_rows = load_jsonl(args.b5_jsonl)
    b5_manifest_path = Path(f"{args.b5_jsonl}.lazy-manifest.json")
    b5_manifest = load_jsonl(b5_manifest_path)[0]
    if (
        b5_manifest.get("benchmark") != args.benchmark
        or b5_manifest.get("output_sha256") != _sha256_file(args.b5_jsonl)
        or b5_manifest.get("base_jsonl", {}).get("sha256") != _sha256_file(args.base_jsonl)
        or b5_manifest.get("combined_jsonl", {}).get("sha256")
        != _sha256_file(args.combined_jsonl)
    ):
        raise ValueError("B5 manifest does not match B6 inputs")
    revision = base_manifest["code"]["revision"]
    pairs = validate_and_extract_combined_pairs(
        args.benchmark, base_rows, combined_rows,
        disabled_launch_manifest=base_manifest,
        enabled_launch_manifest=combined_manifest,
        expected_inference_revision=revision,
    )
    rows = {row["_eg_ordinal"]: row for row in base_rows}
    eligible = [
        pair for pair in pairs
        if is_eligible_lazy_pair(pair, rows[pair.ordinal])
    ]
    b5_by_ordinal = {record["source_ordinal"]: record for record in b5_rows}
    if (
        len(b5_by_ordinal) != len(b5_rows)
        or set(b5_by_ordinal) != {pair.ordinal for pair in eligible}
    ):
        raise ValueError("B5 records do not exactly cover B6 eligibility")
    planned_calls = len(eligible) * 3
    if planned_calls > MAX_CONFIRMATION_CALLS:
        raise ValueError("B6 partition exceeds the frozen call budget")
    qwen_artifact = content_manifest(
        args.model_path, allowed_symlink_root=args.model_path.parents[2],
    )
    if qwen_artifact["sha256"] != b5_manifest["qwen_artifact"]["sha256"]:
        raise ValueError("B6 Qwen artifact differs from frozen B5")
    model = _load_model(args.model_path) if eligible else None
    records = [
        _produce_record(
            pair, rows[pair.ordinal], base_manifest,
            b5_by_ordinal[pair.ordinal], model,
        )
        for pair in eligible
    ]
    output_sha256 = _atomic_write(args.output, records)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase13-hr-semantic-option-loss",
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {"path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl)},
        "combined_jsonl": {
            "path": str(args.combined_jsonl), "sha256": _sha256_file(args.combined_jsonl),
        },
        "b5_jsonl": {
            "path": str(args.b5_jsonl), "sha256": _sha256_file(args.b5_jsonl),
            "manifest_file_sha256": _sha256_file(b5_manifest_path),
            "manifest_sha256": canonical_sha256(b5_manifest),
        },
        "qwen_artifact": qwen_artifact,
        "processor_mode": HR_LOSS_PROCESSOR_MODE,
        "fixed_vstar_rule": B5_FIXED_VSTAR_RULE,
        "rules": [
            hr_loss_rule_key(path, confidence)
            for path in sorted(HR_LOSS_MIN_PATH_CONSENSUS)
            for confidence in sorted(HR_LOSS_MIN_CONFIDENCES)
        ],
        "records": len(records),
        "planned_calls": planned_calls,
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase13_hr_semantic_option_loss.py")
        ),
        "output_sha256": output_sha256,
    }
    _atomic_write(Path(f"{args.output}.hr-loss-manifest.json"), [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
