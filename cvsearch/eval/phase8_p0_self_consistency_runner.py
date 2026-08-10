#!/usr/bin/env python3
"""Produce label-blind P0 dual-view self-consistency observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cvsearch.eval.phase4_expand_oracle as phase4
from cvsearch.eval.phase2_oracle import load_jsonl, load_launch_manifest
from cvsearch.eval.phase5_unified_selector import select_unified_state
from cvsearch.eval.phase6_combined_selection import (
    ExtractedCombinedPair,
    validate_and_extract_combined_pairs,
)
from cvsearch.eval.phase7_confirmation_runner import (
    MAX_CONFIRMATION_CALLS,
    _atomic_write,
    _load_model,
    _sha256_file,
    _vstar_observations,
)
from cvsearch.eval.phase7_uncertainty_confirmation import (
    confirmation_prompt_material,
)
from cvsearch.eval.phase8_p0_self_consistency import (
    P0_CONSISTENCY_THRESHOLDS,
    aggregate_p0_view,
    render_p0_native_views,
    select_p0_self_consistency,
)
from cvsearch.evidence_gap.provenance import canonical_sha256


CALLS_PER_RECORD = 6


def is_eligible_p0_pair(
    pair: ExtractedCombinedPair, raw_row: Mapping[str, Any],
) -> bool:
    """Return whether B1 may supplement this v2-P0 logits example."""
    decision = select_unified_state(pair.p0, pair.candidates)
    trace = raw_row.get("method_trace")
    boxes = trace.get("final_boxes") if isinstance(trace, Mapping) else None
    return (
        decision.action == "P0"
        and raw_row.get("answer_type") == "logits_match"
        and isinstance(boxes, (list, tuple))
        and bool(boxes)
    )


def view_dto(
    projection: Mapping[str, Any] | None, *, view_sha256: str | None,
    prompt_sha256: Sequence[str],
) -> dict[str, Any]:
    feasible = bool(projection is not None and projection.get("feasible") is True)
    return {
        "feasible": feasible,
        "output": projection["output"] if feasible else None,
        "stability": (
            {"confidence": projection["confidence"]} if feasible else None
        ),
        "majority_fraction": (
            projection["majority_fraction"] if feasible else None
        ),
        "view_sha256": view_sha256,
        "prompt_sha256": list(prompt_sha256),
    }


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _produce_record(
    pair: ExtractedCombinedPair, raw_row: Mapping[str, Any],
    base_manifest: Mapping[str, Any], model: Any,
) -> dict[str, Any]:
    source, source_audit = phase4._resolve_source_image(raw_row, base_manifest)
    prompt_material = confirmation_prompt_material(
        raw_row["answer_type"], raw_row["question"], raw_row["options"],
    )
    render_audit = None
    focus_observations = None
    context_observations = None
    focus_projection = None
    context_projection = None
    focus_sha256 = None
    context_sha256 = None
    failure = None
    try:
        focus, context, render_audit = render_p0_native_views(
            source, raw_row["method_trace"]["final_boxes"], model,
        )
        focus_sha256 = render_audit["focus_view_sha256"]
        context_sha256 = render_audit["context_view_sha256"]
        focus_observations = _vstar_observations(
            model, focus, prompt_material["prompts"], raw_row["options"],
        )
        context_observations = _vstar_observations(
            model, context, prompt_material["prompts"], raw_row["options"],
        )
        focus_projection = aggregate_p0_view(
            raw_row["options"], focus_observations,
        )
        context_projection = aggregate_p0_view(
            raw_row["options"], context_observations,
        )
    except Exception as error:
        failure = _failure(error)
    focus_dto = view_dto(
        focus_projection, view_sha256=focus_sha256,
        prompt_sha256=prompt_material["prompt_sha256"],
    )
    context_dto = view_dto(
        context_projection, view_sha256=context_sha256,
        prompt_sha256=prompt_material["prompt_sha256"],
    )
    decisions = {
        str(threshold): asdict(select_p0_self_consistency(
            pair.p0, focus_dto, context_dto, threshold=threshold,
        ))
        for threshold in sorted(P0_CONSISTENCY_THRESHOLDS)
    }
    return {
        "source_ordinal": pair.ordinal,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "p0": pair.p0,
        "focus": focus_dto,
        "context": context_dto,
        "focus_projection": focus_projection,
        "context_projection": context_projection,
        "decisions": decisions,
        "render_audit": render_audit,
        "prompt_material": {
            "answer_type": prompt_material["answer_type"],
            "prompt_sha256": prompt_material["prompt_sha256"],
            "options_sha256": prompt_material["options_sha256"],
        },
        "focus_observations": focus_observations,
        "context_observations": context_observations,
        "failure": failure,
        "cost": {
            "planned_calls": CALLS_PER_RECORD,
            "charged_calls": CALLS_PER_RECORD,
            "charged_source_pixels": (
                CALLS_PER_RECORD * source.width * source.height
            ),
        },
        "source_image": source_audit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
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
    rows_by_ordinal = {row["_eg_ordinal"]: row for row in base_rows}
    eligible = [
        pair for pair in pairs
        if is_eligible_p0_pair(pair, rows_by_ordinal[pair.ordinal])
    ]
    if len(eligible) * CALLS_PER_RECORD > MAX_CONFIRMATION_CALLS:
        raise ValueError("P0 self-consistency partition exceeds the frozen call budget")
    model = _load_model(args.model_path) if eligible else None
    records = [
        _produce_record(
            pair, rows_by_ordinal[pair.ordinal], base_manifest, model,
        )
        for pair in eligible
    ]
    output_sha256 = _atomic_write(args.output, records)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase8-p0-dual-view-self-consistency",
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {
            "path": str(args.base_jsonl),
            "sha256": _sha256_file(args.base_jsonl),
        },
        "combined_jsonl": {
            "path": str(args.combined_jsonl),
            "sha256": _sha256_file(args.combined_jsonl),
        },
        "base_manifest_sha256": canonical_sha256(base_manifest),
        "combined_manifest_sha256": canonical_sha256(combined_manifest),
        "model_artifact_sha256": combined_manifest["artifacts"]["qwen"]["sha256"],
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase8_p0_self_consistency.py")
        ),
        "thresholds": sorted(P0_CONSISTENCY_THRESHOLDS),
        "records": len(records),
        "planned_calls": len(records) * CALLS_PER_RECORD,
        "output_sha256": output_sha256,
    }
    manifest_path = Path(f"{args.output}.self-consistency-manifest.json")
    _atomic_write(manifest_path, [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
