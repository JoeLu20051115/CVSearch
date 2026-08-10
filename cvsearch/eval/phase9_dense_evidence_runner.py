#!/usr/bin/env python3
"""Run independent dense query-evidence search without label access."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from cvsearch.eval.phase7_confirmation_runner import (
    MAX_CONFIRMATION_CALLS,
    _atomic_write,
    _load_model,
    _sha256_file,
)
from cvsearch.eval.phase9_dense_evidence_search import (
    DENSE_MIN_CONFIDENCES,
    DENSE_MIN_GAINS,
    DENSE_TOP_K,
    DenseTile,
    dense_query_texts,
    dense_visual_features,
    generate_dense_tiles,
    project_dense_candidate,
    rank_dense_tiles,
    render_dense_evidence_sheet,
    select_dense_candidate,
)
from cvsearch.evidence_gap.provenance import canonical_sha256, content_manifest


def is_eligible_dense_pair(
    pair: ExtractedCombinedPair, raw_row: Mapping[str, Any],
) -> bool:
    return (
        raw_row.get("answer_type") in {"logits_match", "option_list"}
        and select_unified_state(pair.p0, pair.candidates).action == "P0"
    )


def rule_key(min_confidence: float, min_gain: float) -> str:
    if (
        type(min_confidence) not in (int, float)
        or float(min_confidence) not in DENSE_MIN_CONFIDENCES
        or type(min_gain) not in (int, float)
        or float(min_gain) not in DENSE_MIN_GAINS
    ):
        raise ValueError("dense rule key must use the frozen grid")
    return f"c{float(min_confidence)}-g{float(min_gain)}"


def candidate_dto(
    projection: Mapping[str, Any] | None, *,
    sheet_sha256: Sequence[str | None], rank_sha256: str,
) -> dict[str, Any]:
    feasible = bool(projection is not None and projection.get("feasible") is True)
    return {
        "action": "DENSE",
        "feasible": feasible,
        "output": projection["output"] if feasible else None,
        "stability": (
            {"confidence": projection["confidence"]} if feasible else None
        ),
        "tile_majority_fraction": (
            projection["tile_majority_fraction"] if feasible else None
        ),
        "sheet_sha256": list(sheet_sha256),
        "rank_sha256": rank_sha256,
    }


def _calls_per_record(answer_type: str) -> int:
    return DENSE_TOP_K if answer_type == "logits_match" else DENSE_TOP_K * 4


def _load_clip(path: Path):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(
        path, local_files_only=True, use_fast=False,
    )
    model = CLIPModel.from_pretrained(
        path, local_files_only=True, dtype=torch.float16,
    ).to("cuda:0")
    model.eval()
    return model, processor


@torch.inference_mode()
def _clip_relevance(
    model: Any, processor: Any, source, tiles: Sequence[DenseTile],
    question: str,
) -> list[float]:
    import torch.nn.functional as functional

    text_inputs = processor(
        text=list(dense_query_texts(question)), return_tensors="pt", padding=True,
    ).to("cuda:0")
    image_inputs = processor(
        images=[source.crop(tile.box) for tile in tiles], return_tensors="pt",
    ).to("cuda:0")
    text_features = model.get_text_features(**text_inputs)
    image_features = model.get_image_features(**image_inputs)
    text_features = functional.normalize(text_features.float(), dim=-1)
    image_features = functional.normalize(image_features.float(), dim=-1)
    query_feature = functional.normalize(text_features.mean(dim=0), dim=0)
    values = image_features @ query_feature
    result = [float(value) for value in values.detach().cpu()]
    if len(result) != len(tiles):
        raise ValueError("CLIP relevance did not cover every dense tile")
    return result


def _rank_material(ranked) -> list[dict[str, Any]]:
    return [
        {
            "tile": {
                "identity": item.tile.identity,
                "grid": item.tile.grid,
                "row": item.tile.row,
                "col": item.tile.col,
                "box": list(item.tile.box),
            },
            "relevance": item.relevance,
            "edge_density": item.edge_density,
            "variance": item.variance,
            "relevance_percentile": item.relevance_percentile,
            "edge_percentile": item.edge_percentile,
            "variance_percentile": item.variance_percentile,
            "visual_information": item.visual_information,
            "score": item.score,
        }
        for item in ranked
    ]


def _prepare_rankings(
    pairs: Sequence[ExtractedCombinedPair], rows_by_ordinal: Mapping[int, Mapping[str, Any]],
    base_manifest: Mapping[str, Any], clip_model: Any, clip_processor: Any,
) -> list[dict[str, Any]]:
    prepared = []
    for pair in pairs:
        row = rows_by_ordinal[pair.ordinal]
        source, source_audit = phase4._resolve_source_image(row, base_manifest)
        tiles = generate_dense_tiles(source.width, source.height)
        relevance = _clip_relevance(
            clip_model, clip_processor, source, tiles, row["question"],
        )
        edges, variances = dense_visual_features(source, tiles)
        ranked = rank_dense_tiles(tiles, relevance, edges, variances)
        material = _rank_material(ranked)
        prepared.append({
            "pair": pair,
            "rank_material": material,
            "rank_sha256": canonical_sha256(material),
            "top_tiles": tuple(item.tile for item in ranked[:DENSE_TOP_K]),
            "query_sha256": [
                hashlib.sha256(text.encode()).hexdigest()
                for text in dense_query_texts(row["question"])
            ],
            "source_audit": source_audit,
        })
    return prepared


def _answer_tile(model: Any, row: Mapping[str, Any], sheet) -> Any:
    if row["answer_type"] == "logits_match":
        winner, losses = model.multiple_choices_with_losses(
            sheet, row["question"], row["options"], [],
        )
        return {
            "winner": int(winner),
            "losses": [float(value) for value in losses],
        }
    return [
        model.free_form_using_nodes(
            sheet,
            f"{row['question']}\n{option_block}Answer the option letter directly.",
            [],
        )
        for option_block in row["options"]
    ]


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _produce_record(
    prepared: Mapping[str, Any], row: Mapping[str, Any],
    base_manifest: Mapping[str, Any], model: Any,
) -> dict[str, Any]:
    pair = prepared["pair"]
    source, source_audit = phase4._resolve_source_image(row, base_manifest)
    if source_audit["rgb_pixel_sha256"] != prepared["source_audit"]["rgb_pixel_sha256"]:
        raise ValueError("dense source changed after CLIP ranking")
    sheet_audits = []
    observations = []
    projection = None
    failure = None
    try:
        for tile in prepared["top_tiles"]:
            sheet, audit = render_dense_evidence_sheet(source, tile)
            sheet_audits.append(audit)
            observations.append(_answer_tile(model, row, sheet))
        projection = project_dense_candidate(
            row["answer_type"], row["options"], observations,
        )
    except Exception as error:
        failure = _failure(error)
    sheet_hashes = [audit["sheet_sha256"] for audit in sheet_audits]
    sheet_hashes.extend([None] * (DENSE_TOP_K - len(sheet_hashes)))
    candidate = candidate_dto(
        projection, sheet_sha256=sheet_hashes,
        rank_sha256=prepared["rank_sha256"],
    )
    decisions = {
        rule_key(confidence, gain): asdict(select_dense_candidate(
            pair.p0, candidate,
            min_confidence=confidence, min_gain=gain,
        ))
        for confidence in sorted(DENSE_MIN_CONFIDENCES)
        for gain in sorted(DENSE_MIN_GAINS)
    }
    calls = _calls_per_record(row["answer_type"])
    return {
        "source_ordinal": pair.ordinal,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "p0": pair.p0,
        "candidate": candidate,
        "projection": projection,
        "decisions": decisions,
        "rank_material": prepared["rank_material"],
        "query_sha256": prepared["query_sha256"],
        "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
        "options_sha256": canonical_sha256(row["options"]),
        "sheet_audits": sheet_audits,
        "observations": observations,
        "failure": failure,
        "cost": {
            "planned_calls": calls,
            "charged_calls": calls,
            "charged_source_pixels": calls * source.width * source.height,
        },
        "source_image": source_audit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
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
        if is_eligible_dense_pair(pair, rows_by_ordinal[pair.ordinal])
    ]
    planned_calls = sum(
        _calls_per_record(rows_by_ordinal[pair.ordinal]["answer_type"])
        for pair in eligible
    )
    if planned_calls > MAX_CONFIRMATION_CALLS:
        raise ValueError("dense partition exceeds the frozen Qwen call budget")
    clip_manifest = content_manifest(
        args.clip_model_path,
        allowed_symlink_root=args.clip_model_path.parents[2],
    )
    clip_model, clip_processor = _load_clip(args.clip_model_path)
    prepared = _prepare_rankings(
        eligible, rows_by_ordinal, base_manifest, clip_model, clip_processor,
    )
    del clip_model, clip_processor
    gc.collect()
    torch.cuda.empty_cache()
    model = _load_model(args.model_path) if eligible else None
    records = [
        _produce_record(
            item, rows_by_ordinal[item["pair"].ordinal], base_manifest, model,
        )
        for item in prepared
    ]
    output_sha256 = _atomic_write(args.output, records)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase9-dense-query-evidence-search",
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {
            "path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl),
        },
        "combined_jsonl": {
            "path": str(args.combined_jsonl),
            "sha256": _sha256_file(args.combined_jsonl),
        },
        "base_manifest_sha256": canonical_sha256(base_manifest),
        "combined_manifest_sha256": canonical_sha256(combined_manifest),
        "qwen_artifact_sha256": combined_manifest["artifacts"]["qwen"]["sha256"],
        "clip_artifact": {
            "path": clip_manifest["path"], "sha256": clip_manifest["sha256"],
        },
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase9_dense_evidence_search.py")
        ),
        "rules": [
            rule_key(confidence, gain)
            for confidence in sorted(DENSE_MIN_CONFIDENCES)
            for gain in sorted(DENSE_MIN_GAINS)
        ],
        "records": len(records),
        "planned_calls": planned_calls,
        "output_sha256": output_sha256,
    }
    manifest_path = Path(f"{args.output}.dense-manifest.json")
    _atomic_write(manifest_path, [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
