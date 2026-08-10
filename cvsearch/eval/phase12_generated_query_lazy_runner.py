#!/usr/bin/env python3
"""Run answer-free generated-query ranking with lazy split and backtrack."""

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
from cvsearch.eval.phase9_dense_evidence_runner import _load_clip
from cvsearch.eval.phase9_dense_evidence_search import (
    DenseTile,
    dense_visual_features,
    render_dense_evidence_sheet,
)
from cvsearch.eval.phase12_generated_query_lazy_search import (
    LAZY_GRID,
    LAZY_MIN_CONFIDENCES,
    LAZY_MIN_GAINS,
    LAZY_MIN_PATH_CONSENSUS,
    AdaptivePatch,
    choose_backtrack_patch,
    fuse_query_relevance,
    generate_lazy_children,
    lazy_rule_key,
    lazy_view_record,
    parse_localization_queries,
    project_lazy_candidate,
    rank_lazy_siblings,
    select_lazy_candidate,
)
from cvsearch.evidence_gap.provenance import canonical_sha256, content_manifest


LOCALIZATION_QUERY_PROMPT_VERSION = "answer_free_localization_loc4_v1"
LOCALIZATION_QUERY_TEMPLATE = (
    "Generate exactly four short visual localization phrases for finding the "
    "directly visible evidence needed by this question. Preserve relevant "
    "objects, attributes, and spatial relations. Do not answer the question "
    "and do not infer unseen details. Write one phrase per line and begin every "
    "line with LOC:.\nQuestion: {question}"
)
LOCALIZATION_QUERY_TEMPLATE_SHA256 = hashlib.sha256(
    LOCALIZATION_QUERY_TEMPLATE.encode("utf-8")
).hexdigest()
LAZY_PROCESSOR_MODE = "text_only_qaug_main_top3_parent_child_conditional_root_backtrack"


def is_eligible_lazy_pair(
    pair: ExtractedCombinedPair, raw_row: Mapping[str, Any],
) -> bool:
    return (
        raw_row.get("answer_type") in {"logits_match", "option_list"}
        and select_unified_state(pair.p0, pair.candidates).action == "P0"
    )


def _calls_per_record(answer_type: str) -> int:
    if answer_type == "logits_match":
        return 1 + 3
    if answer_type == "option_list":
        return 1 + 3 * 4
    raise ValueError("lazy answer type is unsupported")


@torch.inference_mode()
def generate_localization_queries(model: Any, question: str) -> dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("localization question must be nonempty")
    user_prompt = LOCALIZATION_QUERY_TEMPLATE.format(question=question)
    chat_prompt = model.get_prompt_from_qs(user_prompt)
    started = time.perf_counter()
    inputs = model.processor(
        text=[chat_prompt], images=None, return_tensors="pt", padding=True,
        padding_side="left",
    ).to(model.device)
    generated = model.model.generate(
        **inputs, use_cache=True, max_new_tokens=128, do_sample=False,
    )
    trimmed = [
        output[len(input_ids):]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    raw = model.processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    queries = parse_localization_queries(raw)
    return {
        "queries": list(queries),
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(user_prompt.encode()).hexdigest(),
        "chat_prompt_sha256": hashlib.sha256(chat_prompt.encode()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def _clip_patch_relevance(
    model: Any, processor: Any, source: Any,
    patches: Sequence[AdaptivePatch], question: str,
    localization_queries: Sequence[str],
) -> list[float]:
    import torch.nn.functional as functional

    if not localization_queries:
        raise ValueError("lazy ranking requires localization queries")
    text = [question, *localization_queries]
    text_inputs = processor(
        text=text, return_tensors="pt", padding=True,
    ).to("cuda:0")
    image_inputs = processor(
        images=[source.crop(patch.box) for patch in patches],
        return_tensors="pt",
    ).to("cuda:0")
    text_features = functional.normalize(
        model.get_text_features(**text_inputs).float(), dim=-1,
    )
    image_features = functional.normalize(
        model.get_image_features(**image_inputs).float(), dim=-1,
    )
    similarities = image_features @ text_features.T
    main = [float(value) for value in similarities[:, 0].detach().cpu()]
    augmented = [
        [float(value) for value in row]
        for row in similarities[:, 1:].detach().cpu()
    ]
    return fuse_query_relevance(main, augmented)


def _rank_material(ranked: Sequence[Any]) -> list[dict[str, Any]]:
    return [{
        "patch": {
            "identity": item.patch.identity,
            "path": list(item.patch.path),
            "box": list(item.patch.box),
        },
        "relevance": item.relevance,
        "edge_density": item.edge_density,
        "variance": item.variance,
        "relevance_percentile": item.relevance_percentile,
        "edge_percentile": item.edge_percentile,
        "variance_percentile": item.variance_percentile,
        "visual_information": item.visual_information,
        "score": item.score,
    } for item in ranked]


def _rank_patch_set(
    clip_model: Any, clip_processor: Any, source: Any,
    patches: Sequence[AdaptivePatch], question: str,
    localization_queries: Sequence[str],
):
    relevance = _clip_patch_relevance(
        clip_model, clip_processor, source, patches, question,
        localization_queries,
    )
    dense = tuple(
        DenseTile(
            LAZY_GRID, patch.path[-1] // LAZY_GRID,
            patch.path[-1] % LAZY_GRID, patch.box,
        )
        for patch in patches
    )
    edges, variances = dense_visual_features(source, dense)
    return rank_lazy_siblings(patches, relevance, edges, variances)


def _render_patch(source: Any, patch: AdaptivePatch) -> tuple[Any, dict[str, Any]]:
    index = patch.path[-1]
    sheet, audit = render_dense_evidence_sheet(source, DenseTile(
        LAZY_GRID, index // LAZY_GRID, index % LAZY_GRID, patch.box,
    ))
    return sheet, {
        **audit,
        "patch_identity": patch.identity,
        "patch_path": list(patch.path),
        "patch_box": list(patch.box),
    }


def _answer_patch(model: Any, row: Mapping[str, Any], sheet: Any) -> Any:
    if row["answer_type"] == "logits_match":
        winner, losses = model.multiple_choices_with_losses(
            sheet, row["question"], row["options"], [],
        )
        return {"winner": int(winner), "losses": [float(value) for value in losses]}
    if row["answer_type"] == "option_list":
        return [
            model.free_form_using_nodes(
                sheet,
                f"{row['question']}\n{block}Answer the option letter directly.",
                [],
            )
            for block in row["options"]
        ]
    raise ValueError("lazy answer type is unsupported")


def _observe_lazy_path(
    model: Any, row: Mapping[str, Any], sheets: Sequence[Any],
) -> tuple[list[Any], bool]:
    if not isinstance(sheets, (list, tuple)) or len(sheets) != 3:
        raise ValueError("lazy observation path requires three available sheets")
    observations = [
        _answer_patch(model, row, sheets[0]),
        _answer_patch(model, row, sheets[1]),
    ]
    first = lazy_view_record(row["answer_type"], row["options"], observations[0])
    second = lazy_view_record(row["answer_type"], row["options"], observations[1])
    agreed = (
        first.aggregation_available is not False
        and second.aggregation_available is not False
        and first.canonical_answer is not None
        and first.canonical_answer == second.canonical_answer
    )
    if agreed:
        return observations, False
    observations.append(_answer_patch(model, row, sheets[2]))
    return observations, True


def _candidate_dto(
    projection: Mapping[str, Any] | None, *, view_sha256: Sequence[str],
    rank_sha256: str, query_sha256: str,
) -> dict[str, Any]:
    feasible = bool(projection is not None and projection.get("feasible") is True)
    return {
        "action": "LAZY",
        "feasible": feasible,
        "output": projection["output"] if feasible else None,
        "stability": {"confidence": projection["confidence"]} if feasible else None,
        "path_consensus": projection["path_consensus"] if feasible else None,
        "view_sha256": list(view_sha256),
        "rank_sha256": rank_sha256,
        "query_sha256": query_sha256,
    }


def _decisions(p0: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        lazy_rule_key(path, confidence, gain): asdict(select_lazy_candidate(
            p0, candidate, min_path_consensus=path,
            min_confidence=confidence, min_gain=gain,
        ))
        for path in sorted(LAZY_MIN_PATH_CONSENSUS)
        for confidence in sorted(LAZY_MIN_CONFIDENCES)
        for gain in sorted(LAZY_MIN_GAINS)
    }


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _produce_record(
    pair: ExtractedCombinedPair, row: Mapping[str, Any],
    base_manifest: Mapping[str, Any], model: Any,
    clip_model: Any, clip_processor: Any,
) -> dict[str, Any]:
    source, source_audit = phase4._resolve_source_image(row, base_manifest)
    query_result = rank_material = render_audits = observations = projection = None
    rank_sha256 = canonical_sha256({"status": "rank_unavailable"})
    query_sha256 = canonical_sha256({"status": "query_unavailable"})
    candidate = None
    failure = None
    used_backtrack = False
    charged_calls = 0
    try:
        charged_calls = 1
        query_result = generate_localization_queries(model, row["question"])
        query_sha256 = canonical_sha256({
            "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
            "prompt_version": LOCALIZATION_QUERY_PROMPT_VERSION,
            "queries": query_result["queries"],
            "raw_response_sha256": query_result["raw_response_sha256"],
            "user_prompt_sha256": query_result["user_prompt_sha256"],
            "chat_prompt_sha256": query_result["chat_prompt_sha256"],
        })
        root = AdaptivePatch((), (0, 0, source.width, source.height))
        roots = generate_lazy_children(root)
        ranked_roots = _rank_patch_set(
            clip_model, clip_processor, source, roots, row["question"],
            query_result["queries"],
        )
        focus = ranked_roots[0].patch
        children = generate_lazy_children(focus)
        ranked_children = _rank_patch_set(
            clip_model, clip_processor, source, children, row["question"],
            query_result["queries"],
        )
        child = ranked_children[0].patch
        backtrack = choose_backtrack_patch(
            tuple(item.patch for item in ranked_roots), child,
            visited={focus.identity},
        )
        rank_material = {
            "root_siblings": _rank_material(ranked_roots),
            "split_siblings": _rank_material(ranked_children),
            "focus_identity": focus.identity,
            "child_identity": child.identity,
            "backtrack_identity": backtrack.identity,
        }
        rank_sha256 = canonical_sha256(rank_material)
        rendered = [_render_patch(source, patch) for patch in (focus, child, backtrack)]
        sheets = [item[0] for item in rendered]
        all_audits = [item[1] for item in rendered]
        observations, used_backtrack = _observe_lazy_path(model, row, sheets)
        render_audits = all_audits[:len(observations)]
        charged_calls += len(observations) * (
            1 if row["answer_type"] == "logits_match" else 4
        )
        projection = project_lazy_candidate(
            row["answer_type"], row["options"], observations,
        )
        candidate = _candidate_dto(
            projection,
            view_sha256=[audit["sheet_sha256"] for audit in render_audits],
            rank_sha256=rank_sha256, query_sha256=query_sha256,
        )
    except Exception as error:
        failure = _failure(error)
        charged_calls = _calls_per_record(row["answer_type"])
    if candidate is None:
        candidate = _candidate_dto(
            None, view_sha256=[], rank_sha256=rank_sha256,
            query_sha256=query_sha256,
        )
    decisions = _decisions(pair.p0, candidate)
    planned_calls = _calls_per_record(row["answer_type"])
    return {
        "source_ordinal": pair.ordinal,
        "input_identity_sha256": canonical_sha256(pair.input_identity),
        "extracted_pair_sha256": pair.extracted_digest,
        "p0": pair.p0,
        "candidate": candidate,
        "decisions": decisions,
        "query_result": query_result,
        "query_sha256": query_sha256,
        "rank_material": rank_material,
        "rank_sha256": rank_sha256,
        "render_audits": render_audits,
        "observations": observations,
        "projection": projection,
        "used_backtrack": used_backtrack,
        "failure": failure,
        "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
        "options_sha256": canonical_sha256(row["options"]),
        "source_image": source_audit,
        "cost": {
            "planned_calls": planned_calls,
            "charged_calls": charged_calls,
            "charged_source_pixels": max(0, charged_calls - 1)
            * source.width * source.height,
        },
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
    rows = {row["_eg_ordinal"]: row for row in base_rows}
    eligible = [
        pair for pair in pairs
        if is_eligible_lazy_pair(pair, rows[pair.ordinal])
    ]
    planned_calls = sum(
        _calls_per_record(rows[pair.ordinal]["answer_type"])
        for pair in eligible
    )
    if planned_calls > MAX_CONFIRMATION_CALLS:
        raise ValueError("lazy partition exceeds the frozen call budget")
    qwen_artifact = content_manifest(
        args.model_path, allowed_symlink_root=args.model_path.parents[2],
    )
    if qwen_artifact["sha256"] != combined_manifest["artifacts"]["qwen"]["sha256"]:
        raise ValueError("lazy Qwen artifact differs from validated inputs")
    clip_artifact = content_manifest(
        args.clip_model_path,
        allowed_symlink_root=args.clip_model_path.parents[2],
    )
    model = _load_model(args.model_path) if eligible else None
    clip_model, clip_processor = _load_clip(args.clip_model_path) if eligible else (None, None)
    records = [
        _produce_record(
            pair, rows[pair.ordinal], base_manifest, model,
            clip_model, clip_processor,
        )
        for pair in eligible
    ]
    output_sha256 = _atomic_write(args.output, records)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase12-generated-query-lazy-search",
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {"path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl)},
        "combined_jsonl": {
            "path": str(args.combined_jsonl), "sha256": _sha256_file(args.combined_jsonl),
        },
        "base_manifest_sha256": canonical_sha256(base_manifest),
        "combined_manifest_sha256": canonical_sha256(combined_manifest),
        "qwen_artifact": qwen_artifact,
        "clip_artifact": clip_artifact,
        "prompt_version": LOCALIZATION_QUERY_PROMPT_VERSION,
        "prompt_template_sha256": LOCALIZATION_QUERY_TEMPLATE_SHA256,
        "processor_mode": LAZY_PROCESSOR_MODE,
        "rules": [
            lazy_rule_key(path, confidence, gain)
            for path in sorted(LAZY_MIN_PATH_CONSENSUS)
            for confidence in sorted(LAZY_MIN_CONFIDENCES)
            for gain in sorted(LAZY_MIN_GAINS)
        ],
        "records": len(records),
        "planned_calls": planned_calls,
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "selector_source_sha256": _sha256_file(
            Path(__file__).with_name("phase12_generated_query_lazy_search.py")
        ),
        "output_sha256": output_sha256,
    }
    _atomic_write(Path(f"{args.output}.lazy-manifest.json"), [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
