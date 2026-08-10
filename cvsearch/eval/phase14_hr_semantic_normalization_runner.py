#!/usr/bin/env python3
"""Replay v2 and apply label-free HR semantic-majority normalization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cvsearch.eval.phase2_oracle import load_jsonl, load_launch_manifest
from cvsearch.eval.phase5_unified_selector import select_unified_state
from cvsearch.eval.phase6_combined_selection import (
    ExtractedCombinedPair,
    validate_and_extract_combined_pairs,
)
from cvsearch.eval.phase7_confirmation_runner import _atomic_write, _sha256_file
from cvsearch.eval.phase14_hr_semantic_normalization import (
    HR_NORMALIZATION_MIN_FREQUENCY,
    project_hr_semantic_normalization,
    select_hr_semantic_normalization,
)
from cvsearch.evidence_gap.provenance import canonical_sha256


def _produce_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
) -> dict[str, Any]:
    v2_decision = select_unified_state(pair.p0, pair.candidates)
    projection = None
    if v2_decision.action == "P0":
        projection = project_hr_semantic_normalization(
            row["options"], pair.p0["output"],
        )
        decision = select_hr_semantic_normalization(pair.p0, row["options"])
    else:
        decision = v2_decision
    return {
        "source_ordinal": pair.ordinal,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "p0": pair.p0,
        "v2_decision": asdict(v2_decision),
        "projection": projection,
        "decision": asdict(decision),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", choices=("hr-bench_4k", "hr-bench_8k"), required=True,
    )
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
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
        args.benchmark, base_rows, combined_rows,
        disabled_launch_manifest=base_manifest,
        enabled_launch_manifest=combined_manifest,
        expected_inference_revision=revision,
    )
    rows = {row["_eg_ordinal"]: row for row in base_rows}
    if len(rows) != len(base_rows) or set(rows) != {pair.ordinal for pair in pairs}:
        raise ValueError("B7 rows do not exactly cover validated v2 pairs")
    records = [_produce_record(pair, rows[pair.ordinal]) for pair in pairs]
    output_sha256 = _atomic_write(args.output, records)
    v2_outputs = [record["v2_decision"]["output"] for record in records]
    selected_outputs = [record["decision"]["output"] for record in records]
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase14-hr-semantic-majority-normalization",
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {
            "path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl),
            "manifest_sha256": canonical_sha256(base_manifest),
        },
        "combined_jsonl": {
            "path": str(args.combined_jsonl),
            "sha256": _sha256_file(args.combined_jsonl),
            "manifest_sha256": canonical_sha256(combined_manifest),
        },
        "selection_rule": {
            "answer_schema": "option_list",
            "requires_v2_action": "P0",
            "minimum_semantic_vote_frequency": HR_NORMALIZATION_MIN_FREQUENCY,
            "requires_unique_projection": True,
        },
        "records": len(records),
        "selected_normalizations": sum(
            record["decision"]["action"] == "HR_NORMALIZE" for record in records
        ),
        "v2_output_sha256": canonical_sha256(v2_outputs),
        "selected_output_sha256": canonical_sha256(selected_outputs),
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase14_hr_semantic_normalization.py")
        ),
        "output_sha256": output_sha256,
    }
    manifest_path = Path(f"{args.output}.normalization-manifest.json")
    _atomic_write(manifest_path, [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
