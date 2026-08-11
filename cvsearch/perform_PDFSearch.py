#!/usr/bin/env python3
"""Resumable runner for the PDF-faithful uncertainty-guided tree search."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

from PIL import Image


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvsearch.evidence_gap.clip_scorer import CLIP_SNAPSHOT
from cvsearch.evidence_gap.input import sanitize_annotation
from cvsearch.evidence_gap.io import JsonlCheckpointWriter
from cvsearch.evidence_gap.pdf_controller import PDFTreeController
from cvsearch.evidence_gap.pdf_runtime import (
    PDFStateEvaluator,
    TreeActionAdapter,
    TreeCatalog,
    absolute_location_geometry,
    build_pdf_query_plan,
    compare_paired_support,
    generate_text_only_response,
    proposed_answer_text,
    wrapper_pairwise_option_probabilities,
    wrapper_yes_no_probability,
)
from cvsearch.evidence_gap.pdf_types import (
    ModelFingerprint,
    PDFSearchConfig,
    validate_independent_checkpoints,
)
from cvsearch.evidence_gap.provenance import (
    build_launch_manifest,
    canonical_sha256,
    content_manifest,
    runtime_environment,
    visible_gpu_uuids,
    write_or_validate_manifest,
)
from cvsearch.evidence_gap.ranking import QueryAwareNodeRanker
from cvsearch.evidence_gap.search_state import SearchStateCollector
from cvsearch.perform_EGSearch import (
    _annotation_file,
    _code_manifest,
    _load_rows,
    _resolve,
    _validate_output,
    parse_ordinals,
    select_annotations,
)


BENCHMARKS = ("vstar", "hr-bench_4k", "hr-bench_8k")
RUNNER_VERSION = "pdf-faithful-v10-cross-node-localization"
PAIR_MIN_AVG_DELTA = 0.1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--generator-model-path", required=True)
    parser.add_argument("--verifier-model-path", required=True)
    parser.add_argument("--annotation-path", "--annotation_path", dest="annotation_path", required=True)
    parser.add_argument("--sam-model-path", required=True)
    parser.add_argument("--nlp-model-path", required=True)
    parser.add_argument("--clip-model-path", default=CLIP_SNAPSHOT)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--generator-device", default="cuda:0")
    parser.add_argument("--verifier-device", default="cuda:0")
    parser.add_argument("--ordinals")
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--split-seed", type=int, default=260809)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def family_candidate_kwargs(family: str) -> dict[str, Any]:
    """Keep the paper's family thresholds while forcing candidate-tree materialization."""
    name = str(family).casefold()
    if name == "llava":
        lower, upper, decay = 0.0, 0.6, [0.1, 0.1, 0.2]
    elif name == "internvl":
        lower, upper, decay = -0.2, 0.2, [0.1, 0.1, 0.2]
    elif name == "qwen":
        lower, upper, decay = 0.0, 0.9, [0.05, 0.1, 0.2]
    else:
        raise ValueError(f"unsupported model family: {family}")
    return {
        "pop_limit": lambda max_depth: max_depth * 3,
        "threshold_descrease": decay,
        "answering_confidence_threshold_lower": lower,
        "answering_confidence_threshold_upper": upper,
        # Confidence wrappers are bounded by one.  Two therefore disables only
        # the legacy quick-return gate and cannot select or label a candidate.
        "fast_threshold": 2.0,
    }


def _strict_json(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON") from error


def _history_spatial_support_count(
    records: Sequence[Mapping[str, Any]], proposal_output: Any,
) -> int:
    """Count distinct focus nodes that stably produce one history proposal."""
    proposal_sha256 = canonical_sha256(proposal_output)
    focus_keys = set()
    for record in records:
        answer = record["answer"]
        if (
            canonical_sha256(answer["output"]) == proposal_sha256
            and answer.get("aggregation_available") is True
            and float(answer.get("frequency", 0.0)) >= 2.0 / 3.0
        ):
            path_keys = record["state"]["path_keys"]
            if not path_keys:
                raise ValueError("history proposal record has no focus path")
            focus_keys.add(path_keys[-1])
    return len(focus_keys)


def _clip_yes_probability(scorer: Any, image: Image.Image, prompt: str) -> float:
    matrix = scorer.score([image], [prompt])
    if not isinstance(matrix, Sequence) or len(matrix) != 1 or len(matrix[0]) != 1:
        raise ValueError("CLIP fallback must return a 1x1 score matrix")
    score = float(matrix[0][0])
    if not math.isfinite(score):
        raise ValueError("CLIP fallback score must be finite")
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, 10.0 * score))))


def _step_dict(step: Any) -> dict[str, Any]:
    return {
        "action": step.action.value,
        "status": step.status,
        "before": step.before.to_dict(),
        "after": step.after.to_dict(),
        "reason": step.reason,
    }


def _controller_dict(result: Any) -> dict[str, Any]:
    return {
        "termination": result.termination.value,
        "selected_history_state_id": result.selected_history_state_id,
        "assessment_count": result.assessment_count,
        "final_state": result.final_state.to_dict(),
        "steps": [_step_dict(step) for step in result.steps],
    }


def _module_activity(adapter: TreeActionAdapter, evaluator: PDFStateEvaluator, result: Any,
                     query_plan: Any, *, safety_fallback_used: bool,
                     paired_reference: Mapping[str, Any]) -> dict[str, Any]:
    changed = Counter(step.action.value for step in result.steps if step.status == "changed")
    no_ops = Counter(step.action.value for step in result.steps if step.status == "no_op")
    support_records = [record["support"] for record in evaluator.records]
    top3 = [
        detail.get("top_k_augmented") == 3
        for detail in adapter.ranking_details
    ]
    return {
        "planner": {
            "structured_success": not query_plan.fallback_used,
            "fallback_used": query_plan.fallback_used,
            "augmented_query_count": len(query_plan.augmented_queries),
        },
        "joint_ranking": {
            "candidate_count": adapter.queue.candidate_count,
            "sibling_groups": adapter.queue.sibling_groups,
            "native_first_choice_changes": adapter.queue.native_first_choice_changes,
            "top3_candidate_count": sum(top3),
            "all_candidates_use_true_top3": bool(top3) and all(top3),
        },
        "uncertainty": {"state_evaluations": len(evaluator.records)},
        "verifier": {
            "independent_state_count": sum(record["independent"] for record in support_records),
            "fallback_state_count": sum(record["fallback_used"] for record in support_records),
            "model_calls": (
                sum(record["model_calls"] for record in support_records)
                + int(paired_reference["extra_model_calls"])
            ),
            "paired_reference_calls": int(paired_reference["extra_model_calls"]),
        },
        "actions": {
            "changed": dict(sorted(changed.items())),
            "no_op": dict(sorted(no_ops.items())),
        },
        "termination": result.termination.value,
        "safety_fallback_used": safety_fallback_used,
    }


def run_pdf_sample(
    *,
    original_annotation: Mapping[str, Any],
    image_folder: str | Path,
    ic_examples: Any,
    config: PDFSearchConfig,
    sam_model: Any,
    generator_model: Any,
    verifier_model: Any,
    nlp_model: Any,
    clip_scorer: Any,
    cvsearch_fn: Any,
    generator_checkpoint_sha256: str,
    verifier_checkpoint_sha256: str,
    generator_family: str = "qwen",
) -> tuple[Any, dict[str, Any]]:
    """Run one label-blind sample and return its answer plus auditable trace."""
    if not isinstance(config, PDFSearchConfig):
        raise TypeError("config must be PDFSearchConfig")
    policy = sanitize_annotation(original_annotation)
    validate_independent_checkpoints(
        ModelFingerprint(generator_family, generator_checkpoint_sha256),
        ModelFingerprint("independent-verifier", verifier_checkpoint_sha256),
    )
    image_value = Path(policy["input_image"]).expanduser()
    image_path = image_value if image_value.is_absolute() else Path(image_folder) / image_value
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

    working = copy.deepcopy(policy)
    collector = SearchStateCollector(image)
    candidate_output = cvsearch_fn(
        sam_model=sam_model,
        zoom_model=generator_model,
        nlp_model=nlp_model,
        annotation=working,
        ic_examples=ic_examples,
        decomposed_question_template="What is the appearance of the {}?",
        image_folder=str(image_folder),
        search_state_sink=collector,
        emit_full_tree_state=True,
        **family_candidate_kwargs(generator_family),
    )
    targets = working.get("targets") or ()
    query_plan = build_pdf_query_plan(
        policy, targets,
        generator=lambda prompt: generate_text_only_response(generator_model, prompt),
    )
    catalog = TreeCatalog.from_collector(collector, image)
    ranking = config.ranking
    ranker = QueryAwareNodeRanker(
        clip_scorer,
        alpha=ranking.alpha,
        beta=ranking.beta,
        visual_lambda=ranking.visual_lambda,
        top_k_augmented=ranking.top_k_augmented,
    )
    adapter = TreeActionAdapter(catalog, image, query_plan, ranker)
    evaluator = PDFStateEvaluator(
        generator_model=generator_model,
        adapter=adapter,
        policy_annotation=policy,
        query_plan=query_plan,
        verifier_probability=lambda view, prompt: wrapper_yes_no_probability(
            verifier_model, view, prompt,
        ),
        fallback_probability=lambda view, prompt: _clip_yes_probability(
            clip_scorer, view, prompt,
        ),
        pair_probability=lambda view, prompt, proposed, reference: (
            wrapper_pairwise_option_probabilities(
                verifier_model, view, prompt, proposed, reference,
            )
        ),
        verifier_checkpoint_sha256=verifier_checkpoint_sha256,
        generator_checkpoint_sha256=generator_checkpoint_sha256,
    )
    result = PDFTreeController(config).run(
        root_key=catalog.root_key,
        assess=evaluator,
        feasible=adapter.feasible,
        execute=adapter.execute,
        branch_available=adapter.branch_available,
    )
    selected_record = next(
        record for record in evaluator.records
        if record["state"]["state_id"] == result.selected_history_state_id
    )
    selected_support = selected_record["support"]
    controller_thresholds = config.controller
    selected_is_supported = (
        selected_support["independent"] is True
        and selected_support["support_avg"] >= controller_thresholds.min_support_avg
        and selected_support["support_min"] >= controller_thresholds.min_support_min
    )
    candidate_sha256 = canonical_sha256(candidate_output)
    candidate_semantic = proposed_answer_text(
        policy["answer_type"], policy["options"], candidate_output,
    )
    proposal_output = result.answer
    proposal_record = selected_record
    proposal_state_id = result.selected_history_state_id
    proposal_origin = "controller"
    if canonical_sha256(result.answer) == candidate_sha256:
        alternatives = []
        for record in evaluator.records:
            output = record["answer"]["output"]
            semantic = proposed_answer_text(
                policy["answer_type"], policy["options"], output,
            )
            if (
                canonical_sha256(output) != candidate_sha256
                and semantic is not None
                and semantic != candidate_semantic
                and record["answer"].get("aggregation_available") is True
                and float(record["answer"].get("frequency", 0.0)) >= 2.0 / 3.0
            ):
                alternatives.append(record)
        if alternatives:
            proposal_record = min(
                alternatives,
                key=lambda record: (
                    float(record["assessment"]["uncertainty"]),
                    -float(record["answer"].get("frequency", 0.0)),
                    int(record["state"]["state_id"]),
                ),
            )
            proposal_output = proposal_record["answer"]["output"]
            proposal_state_id = int(proposal_record["state"]["state_id"])
            proposal_origin = "history_uncertainty_rescue"

    proposal_support = proposal_record["support"]
    state_support_floor = controller_thresholds.min_support_min
    state_support_floor_met = (
        proposal_support["independent"] is True
        and proposal_support["fallback_used"] is False
        and proposal_support["support_min"] >= state_support_floor
    )
    history_spatial_support_count = _history_spatial_support_count(
        evaluator.records, proposal_output,
    )
    history_spatial_consensus_required = (
        proposal_origin == "history_uncertainty_rescue"
    )
    history_spatial_consensus_met = (
        not history_spatial_consensus_required
        or history_spatial_support_count >= 2
    )
    proposal_semantic = proposed_answer_text(
        policy["answer_type"], policy["options"], proposal_output,
    )
    raw_answer_changed = canonical_sha256(proposal_output) != candidate_sha256
    semantic_answers_match = (
        proposal_semantic is not None
        and proposal_semantic == candidate_semantic
    )
    answer_changed = raw_answer_changed and not semantic_answers_match
    proposal_state = evaluator.states[proposal_state_id]
    proposal_geometry = absolute_location_geometry(
        adapter, proposal_state, policy["question"], evaluator.requirements,
    )
    paired_reference: dict[str, Any] = {
        "state_id": proposal_state_id,
        "origin": proposal_origin,
        "geometry": proposal_geometry,
        "required": answer_changed,
        "attempted": False,
        "selected": False,
        "reason": (
            "semantic_answers_match" if raw_answer_changed and semantic_answers_match
            else "answers_match" if not answer_changed
            else "not_attempted"
        ),
        "avg_delta": None,
        "min_delta": None,
        "requirement_wins": None,
        "requirement_count": len(evaluator.requirements),
        "min_avg_delta": PAIR_MIN_AVG_DELTA,
        "min_requirement_delta": 0.0,
        "comparison_mode": None,
        "paraphrase_ids": None,
        "proposed_by_requirement": None,
        "reference_by_requirement": None,
        "state_support": copy.deepcopy(proposal_support),
        "state_support_floor": state_support_floor,
        "state_support_floor_met": state_support_floor_met,
        "history_spatial_consensus_required": history_spatial_consensus_required,
        "history_spatial_support_count": history_spatial_support_count,
        "history_spatial_consensus_met": history_spatial_consensus_met,
        "proposed": None,
        "reference": None,
        "extra_model_calls": 0,
        "extra_processed_pixels": 0,
    }
    if (
        answer_changed
        and proposal_geometry["eligible"]
        and state_support_floor_met
        and history_spatial_consensus_met
    ):
        worst_calls, worst_pixels = evaluator.estimate_paired_support_cost(proposal_state)
        if (
            worst_calls <= result.final_state.remaining_model_calls
            and worst_pixels <= result.final_state.remaining_pixels
        ):
            paired_support = evaluator.verify_paired_output_support(
                proposal_state, proposal_output, candidate_output,
            )
            comparison = compare_paired_support(
                paired_support.proposed,
                paired_support.reference,
                min_avg_delta=PAIR_MIN_AVG_DELTA,
            )
            answer_record = proposal_record["answer"]
            stable = (
                answer_record.get("aggregation_available") is True
                and float(answer_record.get("frequency", 0.0)) >= 2.0 / 3.0
            )
            paired_reference.update(comparison)
            paired_reference.update({
                "attempted": True,
                "selected": comparison["selected"] and stable,
                "comparison_mode": paired_support.mode,
                "paraphrase_ids": list(paired_support.paraphrase_ids),
                "proposed_by_requirement": [
                    list(row) for row in paired_support.proposed_by_requirement
                ],
                "reference_by_requirement": [
                    list(row) for row in paired_support.reference_by_requirement
                ],
                "reason": (
                    "selected_independent_paired_support"
                    if comparison["selected"] and stable
                    else "unstable_answer" if not stable
                    else "paired_support_rejected"
                ),
                "proposed": paired_support.proposed.to_dict(),
                "reference": paired_support.reference.to_dict(),
                "extra_model_calls": paired_support.model_calls,
                "extra_processed_pixels": (
                    paired_support.model_calls
                    * adapter.render_verifier_view(proposal_state).width
                    * adapter.render_verifier_view(proposal_state).height
                ),
            })
        else:
            paired_reference["reason"] = "insufficient_budget"
    elif answer_changed:
        paired_reference["reason"] = (
            "proposal_outside_question_region"
            if not proposal_geometry["absolute_eligible"]
            else "proposal_lacks_relation_enrichment"
            if not proposal_geometry["relation_enriched"]
            else "proposal_lacks_detail_localization"
            if not proposal_geometry["detail_localized"]
            else "proposal_below_state_support_floor"
            if not state_support_floor_met
            else "proposal_lacks_spatial_consensus"
        )

    paired_selected = paired_reference["selected"] is True
    semantic_projection_used = raw_answer_changed and semantic_answers_match
    safety_fallback_used = (
        (answer_changed and not paired_selected)
        or (
            not answer_changed
            and not semantic_projection_used
            and result.termination.value == "FORCED_RETURN"
            and not selected_is_supported
        )
    )
    final_answer = proposal_output if (not answer_changed or paired_selected) else candidate_output
    collector_payload = collector.to_dict()
    trace = {
        "schema_version": 1,
        "method": config.method,
        "profile": config.profile,
        "config": config.to_dict(),
        "generator_family": generator_family,
        "query_plan": query_plan.to_dict(),
        "candidate_factory": {
            "mode": "cvsearch_tree_only_quick_gate_disabled",
            "fast_threshold": 2.0,
            "native_output_sha256": canonical_sha256(candidate_output),
            "root_answer_confidence": working.get("root_ans_conf"),
            "search_mode": working.get("search_mode"),
            "num_pop": copy.deepcopy(working.get("num_pop", [])),
            "truncated_child_edges": catalog.truncated_child_edges,
            "collector_sha256": canonical_sha256(collector_payload),
            "collector": collector_payload,
        },
        "joint_ranking": copy.deepcopy(adapter.ranking_details),
        "state_evaluations": copy.deepcopy(evaluator.records),
        "controller": _controller_dict(result),
        "final_decision": {
            "source": (
                "history_paired_reference"
                if paired_selected and proposal_origin == "history_uncertainty_rescue"
                else "controller_paired_reference" if paired_selected
                else "cvsearch_safety_fallback" if safety_fallback_used
                else "controller"
            ),
            "reason": (
                paired_reference["reason"] if answer_changed
                else "semantic_answers_match" if semantic_projection_used
                else "forced_return_without_independent_support"
                if safety_fallback_used else "controller_answer_retained"
            ),
            "controller_answer_sha256": canonical_sha256(result.answer),
            "proposal_answer_sha256": canonical_sha256(proposal_output),
            "output_sha256": canonical_sha256(final_answer),
            "paired_reference": paired_reference,
        },
        "module_activity": _module_activity(
            adapter, evaluator, result, query_plan,
            safety_fallback_used=safety_fallback_used,
            paired_reference=paired_reference,
        ),
    }
    return _strict_json(final_answer, "PDF answer"), _strict_json(trace, "PDF trace")


def _load_config(path: Path) -> PDFSearchConfig:
    if path.is_symlink() or not path.is_file():
        raise ValueError("reproducible PDF runs require a regular strict JSON config file")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return PDFSearchConfig.from_mapping(value)


def _model_store_root(path: Path) -> Path | None:
    return path.parent.parent if path.parent.name == "snapshots" else None


def _runtime(generator_path: Path, verifier_path: Path, sam_path: Path, nlp_path: Path,
             clip_path: Path, generator_device: str, verifier_device: str) -> tuple[Any, ...]:
    import spacy

    cvsearch_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(cvsearch_dir))
    from CVSearch import get_cvsearch_response
    from cvsearch.evidence_gap.clip_scorer import ClipScorer
    from cvsearch.models.modeling_dispatch import detect_model_family, load_search_model
    from cvsearch.models.modeling_sam3 import sam3_inference

    family = detect_model_family(generator_path)
    generator = load_search_model(generator_path, device=generator_device)
    verifier = load_search_model(verifier_path, device=verifier_device)
    sam = sam3_inference(model_path=str(sam_path))
    nlp = spacy.load(name=str(nlp_path))
    clip = ClipScorer(device=generator_device, model_path=str(clip_path))
    return family, generator, verifier, sam, nlp, clip, get_cvsearch_response


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.force:
        raise ValueError("--resume and --force are mutually exclusive")
    root = Path(args.root_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {
        "generator": _resolve(root, args.generator_model_path),
        "verifier": _resolve(root, args.verifier_model_path),
        "annotations": _resolve(root, args.annotation_path),
        "sam": _resolve(root, args.sam_model_path),
        "nlp": _resolve(root, args.nlp_model_path),
        "clip": _resolve(root, args.clip_model_path),
        "config": _resolve(root, args.config),
    }
    for name in ("generator", "verifier", "annotations", "nlp", "clip"):
        if not paths[name].exists():
            raise FileNotFoundError(paths[name])
    if not paths["sam"].is_file():
        raise FileNotFoundError(paths["sam"])
    config = _load_config(paths["config"])
    ordinals = parse_ordinals(args.ordinals)
    annotation_file, image_benchmark = _annotation_file(paths["annotations"], args.benchmark)
    rows = _load_rows(annotation_file)
    selected = select_annotations(
        rows, args.benchmark, ordinals, args.split, seed=args.split_seed,
        num_chunks=args.num_chunks, chunk_idx=args.chunk_idx,
    )
    expected = tuple(ordinal for ordinal, _ in selected)
    image_folder = paths["annotations"] / image_benchmark
    source_images = []
    seen = set()
    for _, row in selected:
        value = Path(row["input_image"]).expanduser()
        path = value if value.is_absolute() else image_folder / value
        key = str(path.absolute())
        if key not in seen:
            seen.add(key)
            source_images.append(path)
    ic_path = Path(__file__).resolve().parent / "ic_examples" / f"{args.benchmark}.json"
    with ic_path.open("r", encoding="utf-8") as handle:
        ic_examples = json.load(handle)

    code_manifest = _code_manifest()
    source_root = Path(__file__).resolve().parents[1]
    for own_relative in (
        Path("cvsearch/perform_PDFSearch.py"),
        Path("cvsearch/eval/pdf_scores.py"),
        Path("cvsearch/eval/pdf_trace_audit.py"),
    ):
        own_path = source_root / own_relative
        if all(item["path"] != own_relative.as_posix() for item in code_manifest):
            code_manifest.append({
                "path": own_relative.as_posix(),
                "sha256": hashlib.sha256(own_path.read_bytes()).hexdigest(),
            })
    code_manifest.sort(key=lambda item: item["path"])
    code_revision = canonical_sha256(code_manifest)
    launch_manifest = build_launch_manifest(
        benchmark=args.benchmark,
        code_revision=code_revision,
        code_manifest={"files": code_manifest},
        loaded_config=config.to_dict(),
        config_source=paths["config"],
        annotation_file=annotation_file,
        selected_rows=selected,
        source_images=source_images,
        ic_examples=ic_path,
        model_path=paths["generator"],
        sam_path=paths["sam"],
        spacy_path=paths["nlp"],
        # Bind Hugging Face CLIP below with an explicit snapshot-store boundary.
        clip_path=None,
        split=args.split,
        split_seed=args.split_seed,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        environment=runtime_environment(),
        gpu_uuids=visible_gpu_uuids(),
    )
    generator_manifest = launch_manifest["artifacts"].pop("qwen")
    launch_manifest["artifacts"]["generator"] = generator_manifest
    launch_manifest["artifacts"]["verifier"] = content_manifest(
        paths["verifier"], allowed_symlink_root=_model_store_root(paths["verifier"]),
    )
    launch_manifest["artifacts"]["clip"] = content_manifest(
        paths["clip"], allowed_symlink_root=_model_store_root(paths["clip"]),
    )
    launch_manifest["runner_version"] = RUNNER_VERSION
    launch_manifest["devices"] = {
        "generator": args.generator_device,
        "verifier": args.verifier_device,
    }
    fingerprint = canonical_sha256(launch_manifest)
    validate_independent_checkpoints(
        ModelFingerprint("generator", generator_manifest["sha256"]),
        ModelFingerprint("verifier", launch_manifest["artifacts"]["verifier"]["sha256"]),
    )

    answers_path = Path(args.answers_file).expanduser().resolve()
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    write_or_validate_manifest(
        Path(f"{answers_path}.launch-manifest.json"), launch_manifest,
        resume=args.resume, force=args.force,
    )
    writer = JsonlCheckpointWriter(
        answers_path, expected, resume=args.resume, run_fingerprint=fingerprint,
        allow_replace=args.force, code_revision=code_revision,
    )
    try:
        missing = [(ordinal, row) for ordinal, row in selected if ordinal not in writer.completed]
        if not missing:
            writer.finalize()
            return 0
        family, generator, verifier, sam, nlp, clip, cvsearch_fn = _runtime(
            paths["generator"], paths["verifier"], paths["sam"], paths["nlp"],
            paths["clip"], args.generator_device, args.verifier_device,
        )
        for ordinal, original in missing:
            policy = sanitize_annotation(original)
            output, trace = run_pdf_sample(
                original_annotation=original,
                image_folder=image_folder,
                ic_examples=ic_examples,
                config=config,
                sam_model=sam,
                generator_model=generator,
                verifier_model=verifier,
                nlp_model=nlp,
                clip_scorer=clip,
                cvsearch_fn=cvsearch_fn,
                generator_checkpoint_sha256=generator_manifest["sha256"],
                verifier_checkpoint_sha256=launch_manifest["artifacts"]["verifier"]["sha256"],
                generator_family=family,
            )
            _validate_output(args.benchmark, policy, output)
            record = copy.deepcopy(dict(original))
            if "pdf_trace" in record:
                raise ValueError("original annotation contains reserved pdf_trace")
            record["output"] = output
            record["pdf_trace"] = trace
            writer.write(ordinal, _strict_json(record, "PDF output record"))
        writer.finalize()
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
