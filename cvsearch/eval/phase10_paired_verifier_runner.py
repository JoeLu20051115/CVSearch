#!/usr/bin/env python3
"""Run an independent paired answer-support verifier without label access."""

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
from cvsearch.eval.phase9_dense_evidence_runner import (
    candidate_dto,
    is_eligible_dense_pair,
    rule_key as dense_rule_key,
)
from cvsearch.eval.phase9_dense_evidence_search import (
    DENSE_MIN_CONFIDENCES,
    DENSE_MIN_GAINS,
    DENSE_TOP_K,
    DenseTile,
    project_dense_candidate,
    render_dense_evidence_sheet,
    select_dense_candidate,
)
from cvsearch.eval.phase10_paired_verifier import (
    PAIRED_AVG_MARGINS,
    PAIRED_MIN_MARGINS,
    PAIRED_SUPPORT_PROCESSOR_MODE,
    PAIRED_SUPPORT_PROMPT_VERSION,
    PAIRED_SUPPORT_TEMPLATE,
    PairedVerifierDecision,
    paired_rule_key,
    proposed_answer_text,
    select_paired_candidate,
    verifier_prompt,
)
from cvsearch.evidence_gap.provenance import canonical_sha256, content_manifest
from cvsearch.evidence_gap.types import (
    EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE,
    EVIDENCE_SUPPORT_TRANSFORM,
)
from cvsearch.models.modeling_qwenvl import _support_view_sha256


PAIRED_CALLS_PER_RECORD = 6
PAIRED_SUPPORT_TEMPLATE_SHA256 = hashlib.sha256(
    PAIRED_SUPPORT_TEMPLATE.encode("utf-8")
).hexdigest()


@torch.no_grad()
def answer_support_probability(
    model: Any, rendered_observation: Any, *,
    question: str, proposed_answer: str,
) -> dict[str, Any]:
    yes_tokens, no_tokens, yes_id, no_id = model._support_token_provenance()
    fingerprint, checkpoint = model._support_processor_fingerprint()
    prompt = verifier_prompt(question, proposed_answer)
    chat_prompt = model.get_prompt_from_qs("<image>\n" + prompt)
    started = time.perf_counter()
    inputs = model.processor(
        text=[chat_prompt], images=[rendered_observation], return_tensors="pt",
        padding=True, padding_side="left",
    ).to(model.device)
    outputs = model.model(**inputs)
    pair = outputs.logits[0, -1, [yes_id, no_id]]
    if pair.numel() != 2:
        raise ValueError("paired support logits must contain exactly Yes and No")
    probabilities = torch.softmax(pair, dim=-1)
    yes_logit, no_logit = (float(value) for value in pair.detach().cpu())
    p_yes, p_no = (float(value) for value in probabilities.detach().cpu())
    if not all(math.isfinite(value) for value in (yes_logit, no_logit, p_yes, p_no)):
        raise ValueError("paired support logits and probabilities must be finite")
    if abs((p_yes + p_no) - 1.0) > EVIDENCE_SUPPORT_NORMALIZATION_TOLERANCE:
        raise ValueError("paired support probabilities must be normalized")
    return {
        "prompt_version": PAIRED_SUPPORT_PROMPT_VERSION,
        "prompt_template_sha256": PAIRED_SUPPORT_TEMPLATE_SHA256,
        "prompt_sha256": hashlib.sha256(chat_prompt.encode()).hexdigest(),
        "processor_mode": PAIRED_SUPPORT_PROCESSOR_MODE,
        "processor_fingerprint_sha256": canonical_sha256(fingerprint),
        "checkpoint": checkpoint,
        "yes_tokenization": list(yes_tokens),
        "no_tokenization": list(no_tokens),
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "p_yes_transform": EVIDENCE_SUPPORT_TRANSFORM,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "p_yes": p_yes,
        "p_no": p_no,
        "view_sha256": _support_view_sha256(rendered_observation),
        "elapsed_seconds": time.perf_counter() - started,
    }


def record_answer_texts(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    dense_record: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    return (
        proposed_answer_text(row["answer_type"], row["options"], pair.p0["output"]),
        proposed_answer_text(
            row["answer_type"], row["options"], dense_record["candidate"]["output"],
        ),
    )


def is_verifier_candidate(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    dense_record: Mapping[str, Any],
) -> bool:
    candidate = dense_record.get("candidate")
    return (
        row.get("answer_type") in {"logits_match", "option_list"}
        and dense_record.get("p0") == pair.p0
        and isinstance(candidate, Mapping)
        and candidate.get("feasible") is True
        and candidate.get("output") != pair.p0["output"]
        and select_unified_state(pair.p0, pair.candidates).action == "P0"
    )


def paired_decisions(
    p0: dict[str, Any], candidate: dict[str, Any], *,
    p0_support: Any, candidate_support: Any,
) -> dict[str, dict[str, Any]]:
    return {
        paired_rule_key(avg_margin, min_margin): asdict(select_paired_candidate(
            p0, candidate,
            p0_support=p0_support, candidate_support=candidate_support,
            min_avg_delta=avg_margin, min_min_delta=min_margin,
        ))
        for avg_margin in sorted(PAIRED_AVG_MARGINS)
        for min_margin in sorted(PAIRED_MIN_MARGINS)
    }


def _unavailable_decisions(p0: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    decision = asdict(PairedVerifierDecision(
        "P0", "verifier_unavailable", p0["output"], None, None, None,
    ))
    return {
        paired_rule_key(avg_margin, min_margin): dict(decision)
        for avg_margin in sorted(PAIRED_AVG_MARGINS)
        for min_margin in sorted(PAIRED_MIN_MARGINS)
    }


def _validate_dense_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any], record: Mapping[str, Any],
) -> None:
    if (
        record.get("source_ordinal") != pair.ordinal
        or record.get("input_identity_sha256") != canonical_sha256(pair.input_identity)
        or record.get("extracted_pair_sha256") != pair.extracted_digest
        or record.get("p0") != pair.p0
        or record.get("question_sha256")
        != hashlib.sha256(row["question"].encode()).hexdigest()
        or record.get("options_sha256") != canonical_sha256(row["options"])
    ):
        raise ValueError("B2 dense record does not bind its validated input pair")
    rank_sha256 = canonical_sha256(record.get("rank_material"))
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("rank_sha256") != rank_sha256:
        raise ValueError("B2 dense candidate does not bind its rank material")
    audits = record.get("sheet_audits")
    if not isinstance(audits, list) or len(audits) > DENSE_TOP_K:
        raise ValueError("B2 dense record has invalid sheet audits")
    sheet_hashes = [audit["sheet_sha256"] for audit in audits]
    sheet_hashes.extend([None] * (DENSE_TOP_K - len(sheet_hashes)))
    projection = None
    if record.get("failure") is None:
        projection = project_dense_candidate(
            row["answer_type"], row["options"], record.get("observations"),
        )
    rebuilt = candidate_dto(
        projection, sheet_sha256=sheet_hashes, rank_sha256=rank_sha256,
    )
    if rebuilt != candidate or record.get("projection") != projection:
        raise ValueError("B2 dense candidate projection cannot be reconstructed")
    decisions = {
        dense_rule_key(confidence, gain): asdict(select_dense_candidate(
            pair.p0, rebuilt, min_confidence=confidence, min_gain=gain,
        ))
        for confidence in sorted(DENSE_MIN_CONFIDENCES)
        for gain in sorted(DENSE_MIN_GAINS)
    }
    if decisions != record.get("decisions"):
        raise ValueError("B2 dense decisions cannot be reconstructed")


def _render_bound_sheets(
    row: Mapping[str, Any], base_manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[Any]:
    source, source_audit = phase4._resolve_source_image(row, base_manifest)
    if source_audit != record.get("source_image"):
        raise ValueError("B2 source image changed before paired verification")
    rank_tiles = {
        item["tile"]["identity"]: item["tile"]
        for item in record["rank_material"]
    }
    sheets = []
    for expected in record["sheet_audits"]:
        material = rank_tiles.get(expected["tile_identity"])
        if material is None:
            raise ValueError("B2 sheet tile is absent from rank material")
        tile = DenseTile(
            material["grid"], material["row"], material["col"],
            tuple(material["box"]),
        )
        sheet, audit = render_dense_evidence_sheet(source, tile)
        if audit != expected:
            raise ValueError("B2 evidence sheet changed before paired verification")
        sheets.append(sheet)
    if len(sheets) != DENSE_TOP_K:
        raise ValueError("paired verifier requires the exact three B2 sheets")
    return sheets


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _produce_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    base_manifest: Mapping[str, Any], dense_record: Mapping[str, Any], model: Any,
) -> dict[str, Any]:
    p0_text, candidate_text = record_answer_texts(pair, row, dense_record)
    support = None
    failure = None
    decisions = _unavailable_decisions(pair.p0)
    charged_calls = 0
    try:
        if p0_text is None or candidate_text is None:
            raise ValueError("paired answer text is unavailable")
        sheets = _render_bound_sheets(row, base_manifest, dense_record)
        charged_calls = PAIRED_CALLS_PER_RECORD
        p0_results = []
        candidate_results = []
        for sheet in sheets:
            p0_results.append(answer_support_probability(
                model, sheet, question=row["question"], proposed_answer=p0_text,
            ))
            candidate_results.append(answer_support_probability(
                model, sheet, question=row["question"],
                proposed_answer=candidate_text,
            ))
        support = {"p0": p0_results, "candidate": candidate_results}
        decisions = paired_decisions(
            pair.p0, dense_record["candidate"],
            p0_support=[item["p_yes"] for item in p0_results],
            candidate_support=[item["p_yes"] for item in candidate_results],
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
        "support": support,
        "decisions": decisions,
        "failure": failure,
        "cost": {
            "planned_calls": PAIRED_CALLS_PER_RECORD,
            "charged_calls": charged_calls,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
    parser.add_argument("--dense-jsonl", type=Path, required=True)
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
    dense_manifest_path = Path(f"{args.dense_jsonl}.dense-manifest.json")
    dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
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
        raise ValueError("B2 dense manifest does not match paired verifier inputs")
    pairs = validate_and_extract_combined_pairs(
        args.benchmark, base_rows, combined_rows,
        disabled_launch_manifest=base_manifest,
        enabled_launch_manifest=combined_manifest,
        expected_inference_revision=revision,
    )
    rows_by_ordinal = {row["_eg_ordinal"]: row for row in base_rows}
    dense_by_ordinal = {row["source_ordinal"]: row for row in dense_rows}
    eligible_ordinals = {
        pair.ordinal for pair in pairs
        if is_eligible_dense_pair(pair, rows_by_ordinal[pair.ordinal])
    }
    if (
        len(dense_by_ordinal) != len(dense_rows)
        or set(dense_by_ordinal) != eligible_ordinals
    ):
        raise ValueError("B2 dense records do not exactly cover eligible pairs")
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
    planned_calls = len(targets) * PAIRED_CALLS_PER_RECORD
    if planned_calls > MAX_CONFIRMATION_CALLS:
        raise ValueError("paired verifier partition exceeds the frozen call budget")
    verifier_artifact = content_manifest(
        args.verifier_model_path,
        allowed_symlink_root=args.verifier_model_path.parents[2],
    )
    model = _load_model(args.verifier_model_path) if targets else None
    records = [
        _produce_record(
            pair, rows_by_ordinal[pair.ordinal], base_manifest,
            dense_by_ordinal[pair.ordinal], model,
        )
        for pair in targets
    ]
    output_sha256 = _atomic_write(args.output, records)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase10-independent-paired-answer-support",
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
        "generator_artifact_sha256": dense_manifest["qwen_artifact_sha256"],
        "verifier_artifact": verifier_artifact,
        "prompt_version": PAIRED_SUPPORT_PROMPT_VERSION,
        "prompt_template_sha256": PAIRED_SUPPORT_TEMPLATE_SHA256,
        "processor_mode": PAIRED_SUPPORT_PROCESSOR_MODE,
        "rules": [
            paired_rule_key(avg_margin, min_margin)
            for avg_margin in sorted(PAIRED_AVG_MARGINS)
            for min_margin in sorted(PAIRED_MIN_MARGINS)
        ],
        "records": len(records),
        "planned_calls": planned_calls,
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase10_paired_verifier.py")
        ),
        "output_sha256": output_sha256,
    }
    _atomic_write(Path(f"{args.output}.paired-manifest.json"), [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
