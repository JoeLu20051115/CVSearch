#!/usr/bin/env python3
"""Produce label-blind cross-view confirmations from validated combined runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import cvsearch.eval.phase4_expand_oracle as phase4
from cvsearch.eval.phase2_oracle import load_jsonl, load_launch_manifest
from cvsearch.eval.phase5_unified_selector import select_unified_state
from cvsearch.eval.phase6_combined_selection import (
    ExtractedCombinedPair,
    validate_and_extract_combined_pairs,
)
from cvsearch.eval.phase7_uncertainty_confirmation import (
    CONFIRMATION_THRESHOLDS,
    aggregate_confirmation_views,
    compose_confirmation_view,
    confirm_action_candidate,
    confirmation_prompt_material,
    render_expand_action_views,
    render_zoom_action_views,
)
from cvsearch.evidence_gap.provenance import canonical_sha256


MAX_CONFIRMATION_CALLS = 512


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_confirmation_actions(
    pair: ExtractedCombinedPair,
) -> tuple[dict[str, Any], ...]:
    """Return changed feasible actions not already admitted by the v2 rule."""
    result = []
    for candidate in pair.candidates:
        decision = select_unified_state(pair.p0, [candidate])
        if (
            candidate["feasible"]
            and candidate["output"] != pair.p0["output"]
            and decision.action == "P0"
        ):
            result.append(candidate)
    return tuple(result)


def _load_model(path: Path):
    from cvsearch.models.modeling_qwenvl import ModelQwenVL

    return ModelQwenVL(
        model_path=str(path), device="cuda:0", torch_dtype=torch.bfloat16,
        patch_scale=1.2,
    )


def _action_plan(row: Mapping[str, Any], action: str) -> Mapping[str, Any]:
    index = {"ZOOM": 0, "EXPAND": 1}[action]
    step = row["method_trace"]["steps"][index]
    audit = step[f"{action.casefold()}_audit"]
    batch = audit["batch_result"]
    if not audit["feasible"] or batch["status"] != "success":
        raise ValueError(f"eligible {action} lacks one successful action batch")
    return batch["batch_plan"]


def _candidate_raw(row: Mapping[str, Any], action: str) -> Any:
    index = {"ZOOM": 0, "EXPAND": 1}[action]
    audit = row["method_trace"]["steps"][index][f"{action.casefold()}_audit"]
    return audit["batch_result"]["candidate_answer"]


def _vstar_observations(model: Any, image, prompts, options) -> list[dict[str, Any]]:
    result = []
    for prompt in prompts:
        winner, losses = model.multiple_choices_with_losses(
            image.copy(), prompt, list(options), [],
        )
        result.append({
            "winner": int(winner),
            "losses": [float(value) for value in losses],
        })
    return result


def _hr_observations(model: Any, image, prompts) -> list[str]:
    return [
        model.free_form_using_nodes(image.copy(), prompt, [])
        for prompt in prompts
    ]


def _infeasible_confirmation(action: str) -> dict[str, Any]:
    return {
        "action": action, "feasible": False, "output": None,
        "stability": None, "aggregate_stability": None,
        "view_sha256": None, "prompt_sha256": None,
    }


def _produce_record(
    pair: ExtractedCombinedPair, raw_row: Mapping[str, Any],
    combined_manifest: Mapping[str, Any], action_candidate: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    action = action_candidate["action"]
    source, source_audit = phase4._resolve_source_image(raw_row, combined_manifest)
    plan = _action_plan(raw_row, action)
    if action == "ZOOM":
        candidate_view, broader_view, action_render = render_zoom_action_views(
            source, plan, model,
        )
    else:
        candidate_view, broader_view, action_render = render_expand_action_views(
            source, plan,
        )
    confirmation_view, confirmation_render = compose_confirmation_view(
        candidate_view, broader=broader_view, source_image=source,
    )
    prompt_material = confirmation_prompt_material(
        raw_row["answer_type"], raw_row["question"], raw_row["options"],
    )
    planned_calls = 6 if raw_row["answer_type"] == "logits_match" else 4
    candidate_observations = None
    confirmation_observations = None
    projection = None
    failure = None
    try:
        if raw_row["answer_type"] == "logits_match":
            candidate_observations = _vstar_observations(
                model, candidate_view, prompt_material["prompts"], raw_row["options"],
            )
            confirmation_observations = _vstar_observations(
                model, confirmation_view, prompt_material["prompts"], raw_row["options"],
            )
        else:
            candidate_observations = _candidate_raw(raw_row, action)
            confirmation_observations = _hr_observations(
                model, confirmation_view, prompt_material["prompts"],
            )
        projection = aggregate_confirmation_views(
            raw_row["answer_type"], raw_row["options"],
            candidate_observations, confirmation_observations,
        )
    except Exception as error:
        failure = {
            "type": type(error).__name__,
            "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        }

    candidate = {
        "action": action,
        "feasible": True,
        "output": action_candidate["output"],
        "stability": action_candidate["candidate_stability"],
        "view_sha256": action_render["candidate_view_sha256"],
    }
    if (
        failure is None and projection is not None and projection["feasible"]
        and len(set(prompt_material["prompt_sha256"]))
        == len(prompt_material["prompt_sha256"])
    ):
        confirmation = {
            "action": action,
            "feasible": True,
            "output": projection["output"],
            "stability": {
                "confidence": projection["confirmation_confidence"],
            },
            "aggregate_stability": {
                "confidence": projection["aggregate_confidence"],
            },
            "view_sha256": confirmation_render["confirmation_view_sha256"],
            "prompt_sha256": prompt_material["prompt_sha256"],
        }
    else:
        confirmation = _infeasible_confirmation(action)
    decisions = {}
    for threshold in sorted(CONFIRMATION_THRESHOLDS):
        decisions[str(threshold)] = asdict(confirm_action_candidate(
            pair.p0, candidate, confirmation, threshold=threshold,
        ))
    return {
        "source_ordinal": pair.ordinal,
        "action": action,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "p0": pair.p0,
        "candidate": candidate,
        "confirmation": confirmation,
        "projection": projection,
        "decisions": decisions,
        "action_render": action_render,
        "confirmation_render": confirmation_render,
        "prompt_material": {
            "answer_type": prompt_material["answer_type"],
            "prompt_sha256": prompt_material["prompt_sha256"],
            "options_sha256": prompt_material["options_sha256"],
        },
        "candidate_observations": candidate_observations,
        "confirmation_observations": confirmation_observations,
        "failure": failure,
        "cost": {
            "planned_calls": planned_calls,
            "charged_calls": planned_calls,
            "charged_source_pixels": planned_calls * source.width * source.height,
        },
        "source_image": source_audit,
    }


def _atomic_write(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    partial = Path(f"{path}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with partial.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(
                    record, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False,
                ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return _sha256_file(path)


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
    eligible = [
        (pair, candidate)
        for pair in pairs for candidate in eligible_confirmation_actions(pair)
    ]
    answer_type = base_rows[0]["answer_type"]
    calls_per_candidate = 6 if answer_type == "logits_match" else 4
    if len(eligible) * calls_per_candidate > MAX_CONFIRMATION_CALLS:
        raise ValueError("confirmation partition exceeds the frozen call budget")
    model = _load_model(args.model_path) if eligible else None
    rows_by_ordinal = {row["_eg_ordinal"]: row for row in combined_rows}
    records = [
        _produce_record(
            pair, rows_by_ordinal[pair.ordinal], combined_manifest, candidate, model,
        )
        for pair, candidate in eligible
    ]
    output_sha256 = _atomic_write(args.output, records)
    manifest = {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {"path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl)},
        "combined_jsonl": {
            "path": str(args.combined_jsonl), "sha256": _sha256_file(args.combined_jsonl),
        },
        "base_manifest_sha256": canonical_sha256(base_manifest),
        "combined_manifest_sha256": canonical_sha256(combined_manifest),
        "model_artifact_sha256": combined_manifest["artifacts"]["qwen"]["sha256"],
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase7_uncertainty_confirmation.py")
        ),
        "records": len(records),
        "planned_calls": len(records) * calls_per_candidate,
        "output_sha256": output_sha256,
    }
    manifest_path = Path(f"{args.output}.confirmation-manifest.json")
    _atomic_write(manifest_path, [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
