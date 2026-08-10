#!/usr/bin/env python3
"""Run the label-blind joint-evidence cross-model answer duel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

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
from cvsearch.eval.phase9_dense_evidence_search import DenseTile
from cvsearch.eval.phase10_paired_verifier_runner import (
    _support_view_sha256,
    _validate_dense_record,
    is_verifier_candidate,
    record_answer_texts,
)
from cvsearch.eval.phase11_joint_evidence_duel import (
    JOINT_DUEL_PROMPT_VERSION,
    JOINT_DUEL_TEMPLATE,
    JointDuelDecision,
    duel_prompts,
    project_joint_duel,
    render_joint_evidence_sheet,
    select_joint_candidate,
)
from cvsearch.evidence_gap.provenance import canonical_sha256, content_manifest


JOINT_CALLS_PER_RECORD = 3
JOINT_DUEL_TEMPLATE_SHA256 = hashlib.sha256(
    JOINT_DUEL_TEMPLATE.encode("utf-8")
).hexdigest()
JOINT_DUEL_PROCESSOR_MODE = (
    "joint_sheet_order_swapped_final_ab_logits_plus_generator_option_loss"
)


@torch.no_grad()
def answer_duel_logits(
    model: Any, rendered_observation: Any, prompt: str,
) -> dict[str, Any]:
    a_tokens = tuple(int(value) for value in model.tokenizer("A").input_ids)
    b_tokens = tuple(int(value) for value in model.tokenizer("B").input_ids)
    if len(a_tokens) != 1 or len(b_tokens) != 1 or a_tokens[0] == b_tokens[0]:
        raise ValueError("A and B must each be one distinct verifier token")
    chat_prompt = model.get_prompt_from_qs("<image>\n" + prompt)
    started = time.perf_counter()
    inputs = model.processor(
        text=[chat_prompt], images=[rendered_observation], return_tensors="pt",
        padding=True, padding_side="left",
    ).to(model.device)
    outputs = model.model(**inputs)
    pair = outputs.logits[0, -1, [a_tokens[0], b_tokens[0]]]
    if pair.numel() != 2:
        raise ValueError("duel logits must contain exactly A and B")
    a_logit, b_logit = (float(value) for value in pair.detach().cpu())
    if not all(math.isfinite(value) for value in (a_logit, b_logit)):
        raise ValueError("duel logits must be finite")
    winner = min((0, 1), key=lambda index: (-(a_logit, b_logit)[index], index))
    return {
        "winner": winner,
        "a_logit": a_logit,
        "b_logit": b_logit,
        "a_tokenization": list(a_tokens),
        "b_tokenization": list(b_tokens),
        "a_token_id": a_tokens[0],
        "b_token_id": b_tokens[0],
        "prompt_sha256": hashlib.sha256(chat_prompt.encode()).hexdigest(),
        "view_sha256": _support_view_sha256(rendered_observation),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _ranked_tiles(record: Mapping[str, Any]) -> tuple[DenseTile, DenseTile, DenseTile]:
    material = {
        item["tile"]["identity"]: item["tile"]
        for item in record["rank_material"]
    }
    result = []
    for audit in record["sheet_audits"]:
        tile = material.get(audit["tile_identity"])
        if tile is None or list(tile["box"]) != audit["tile_box"]:
            raise ValueError("joint duel tile differs from frozen B2 evidence")
        result.append(DenseTile(
            tile["grid"], tile["row"], tile["col"], tuple(tile["box"]),
        ))
    if len(result) != 3:
        raise ValueError("joint duel requires three frozen B2 tiles")
    return tuple(result)  # type: ignore[return-value]


def _render_bound_joint_sheet(
    row: Mapping[str, Any], base_manifest: Mapping[str, Any],
    dense_record: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    source, source_audit = phase4._resolve_source_image(row, base_manifest)
    if source_audit != dense_record.get("source_image"):
        raise ValueError("B2 source changed before joint duel")
    return render_joint_evidence_sheet(source, _ranked_tiles(dense_record))


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _produce_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    base_manifest: Mapping[str, Any], dense_record: Mapping[str, Any],
    generator_model: Any, verifier_model: Any,
) -> dict[str, Any]:
    p0_text, candidate_text = record_answer_texts(pair, row, dense_record)
    render_audit = observations = projection = None
    decision = JointDuelDecision("P0", "duel_unavailable", pair.p0["output"])
    failure = None
    charged_calls = 0
    try:
        if p0_text is None or candidate_text is None:
            raise ValueError("joint duel answer text is unavailable")
        sheet, render_audit = _render_bound_joint_sheet(
            row, base_manifest, dense_record,
        )
        charged_calls = JOINT_CALLS_PER_RECORD
        prompts = duel_prompts(row["question"], p0_text, candidate_text)
        p0_first = answer_duel_logits(
            verifier_model, sheet, prompts["p0_first"],
        )
        candidate_first = answer_duel_logits(
            verifier_model, sheet, prompts["candidate_first"],
        )
        generator_question = (
            f"{row['question']}\nAnswer with the proposed answer best supported "
            "by the visible evidence."
        )
        winner, losses = generator_model.multiple_choices_with_losses(
            sheet, generator_question, [p0_text, candidate_text], [],
        )
        generator = {
            "winner": int(winner), "losses": [float(value) for value in losses],
        }
        observations = {
            "p0_first": p0_first,
            "candidate_first": candidate_first,
            "generator": generator,
            "prompt_sha256": {
                key: hashlib.sha256(value.encode()).hexdigest()
                for key, value in prompts.items()
            },
            "generator_question_sha256": hashlib.sha256(
                generator_question.encode()
            ).hexdigest(),
        }
        projection = project_joint_duel(
            p0_first={key: p0_first[key] for key in ("winner", "a_logit", "b_logit")},
            candidate_first={
                key: candidate_first[key] for key in ("winner", "a_logit", "b_logit")
            },
            generator=generator,
        )
        decision = select_joint_candidate(
            pair.p0, dense_record["candidate"], projection,
        )
    except Exception as error:
        failure = _failure(error)
    return {
        "source_ordinal": pair.ordinal,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "b2_record_sha256": canonical_sha256(dense_record),
        "p0": pair.p0,
        "candidate": dense_record["candidate"],
        "p0_answer_text": p0_text,
        "candidate_answer_text": candidate_text,
        "render_audit": render_audit,
        "observations": observations,
        "projection": projection,
        "decision": asdict(decision),
        "failure": failure,
        "cost": {
            "planned_calls": JOINT_CALLS_PER_RECORD,
            "charged_calls": charged_calls,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
    parser.add_argument("--dense-jsonl", type=Path, required=True)
    parser.add_argument("--generator-model-path", type=Path, required=True)
    parser.add_argument("--verifier-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_rows = load_jsonl(args.base_jsonl)
    combined_rows = load_jsonl(args.combined_jsonl)
    base_manifest = load_launch_manifest(args.base_jsonl)
    combined_manifest = load_launch_manifest(args.combined_jsonl)
    dense_rows = load_jsonl(args.dense_jsonl)
    dense_manifest = json.loads(Path(
        f"{args.dense_jsonl}.dense-manifest.json"
    ).read_text(encoding="utf-8"))
    revision = base_manifest["code"]["revision"]
    if (
        dense_manifest.get("benchmark") != args.benchmark
        or dense_manifest.get("inference_revision") != revision
        or dense_manifest.get("output_sha256") != _sha256_file(args.dense_jsonl)
        or dense_manifest.get("base_jsonl", {}).get("sha256")
        != _sha256_file(args.base_jsonl)
        or dense_manifest.get("combined_jsonl", {}).get("sha256")
        != _sha256_file(args.combined_jsonl)
        or dense_manifest.get("base_manifest_sha256") != canonical_sha256(base_manifest)
        or dense_manifest.get("combined_manifest_sha256")
        != canonical_sha256(combined_manifest)
    ):
        raise ValueError("B2 dense manifest does not match joint duel inputs")
    pairs = validate_and_extract_combined_pairs(
        args.benchmark, base_rows, combined_rows,
        disabled_launch_manifest=base_manifest,
        enabled_launch_manifest=combined_manifest,
        expected_inference_revision=revision,
    )
    rows_by_ordinal = {row["_eg_ordinal"]: row for row in base_rows}
    dense_by_ordinal = {row["source_ordinal"]: row for row in dense_rows}
    if len(dense_by_ordinal) != len(dense_rows):
        raise ValueError("B2 dense records contain duplicate ordinals")
    for pair in pairs:
        if pair.ordinal in dense_by_ordinal:
            _validate_dense_record(
                pair, rows_by_ordinal[pair.ordinal], dense_by_ordinal[pair.ordinal],
            )
    targets = [
        pair for pair in pairs
        if pair.ordinal in dense_by_ordinal and is_verifier_candidate(
            pair, rows_by_ordinal[pair.ordinal], dense_by_ordinal[pair.ordinal],
        )
    ]
    planned_calls = len(targets) * JOINT_CALLS_PER_RECORD
    if planned_calls > MAX_CONFIRMATION_CALLS:
        raise ValueError("joint duel exceeds the frozen call budget")
    generator_artifact = content_manifest(
        args.generator_model_path,
        allowed_symlink_root=args.generator_model_path.parents[2],
    )
    verifier_artifact = content_manifest(
        args.verifier_model_path,
        allowed_symlink_root=args.verifier_model_path.parents[2],
    )
    if (
        generator_artifact["sha256"] != dense_manifest["qwen_artifact_sha256"]
        or generator_artifact["sha256"] == verifier_artifact["sha256"]
    ):
        raise ValueError("joint duel generator/verifier artifacts are not independent")
    generator_model = _load_model(args.generator_model_path) if targets else None
    verifier_model = _load_model(args.verifier_model_path) if targets else None
    records = [
        _produce_record(
            pair, rows_by_ordinal[pair.ordinal], base_manifest,
            dense_by_ordinal[pair.ordinal], generator_model, verifier_model,
        )
        for pair in targets
    ]
    output_sha256 = _atomic_write(args.output, records)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase11-joint-evidence-cross-model-duel",
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {"path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl)},
        "combined_jsonl": {
            "path": str(args.combined_jsonl), "sha256": _sha256_file(args.combined_jsonl),
        },
        "dense_jsonl": {
            "path": str(args.dense_jsonl), "sha256": _sha256_file(args.dense_jsonl),
            "manifest_sha256": canonical_sha256(dense_manifest),
        },
        "generator_artifact": generator_artifact,
        "verifier_artifact": verifier_artifact,
        "prompt_version": JOINT_DUEL_PROMPT_VERSION,
        "prompt_template_sha256": JOINT_DUEL_TEMPLATE_SHA256,
        "processor_mode": JOINT_DUEL_PROCESSOR_MODE,
        "rule": "cross-model-order-balanced-unanimous-v1",
        "records": len(records),
        "planned_calls": planned_calls,
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase11_joint_evidence_duel.py")
        ),
        "output_sha256": output_sha256,
    }
    _atomic_write(Path(f"{args.output}.duel-manifest.json"), [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
