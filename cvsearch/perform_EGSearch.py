#!/usr/bin/env python3
"""Resumable, sanitized runner for the minimal evidence-gap method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvsearch.evidence_gap.clip_scorer import CLIP_SNAPSHOT
from cvsearch.evidence_gap.input import sanitize_annotation, split_bucket
from cvsearch.evidence_gap.io import JsonlCheckpointWriter
from cvsearch.evidence_gap.method import compose_output_record, get_evidence_gap_response, load_method_config


BENCHMARKS = (
    "vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld-lite", "treebench",
    "fines-bench_option", "fines-bench_reasoning",
)
RUNNER_VERSION = "evidence-gap-task7-v1"


def parse_ordinals(spec: str | None) -> tuple[int, ...] | None:
    if spec is None:
        return None
    if not isinstance(spec, str) or not spec:
        raise ValueError("ordinals must be a nonempty comma/range expression")
    values: list[int] = []
    for part in spec.split(","):
        if not part or part.strip() != part:
            raise ValueError(f"invalid ordinal component: {part!r}")
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2 or not all(piece.isdecimal() for piece in pieces):
                raise ValueError(f"invalid ordinal range: {part!r}")
            start, end = map(int, pieces)
            if end < start:
                raise ValueError(f"descending ordinal range: {part!r}")
            values.extend(range(start, end + 1))
        elif part.isdecimal():
            values.append(int(part))
        else:
            raise ValueError(f"invalid ordinal: {part!r}")
    if len(values) != len(set(values)):
        raise ValueError("ordinals must not contain duplicates")
    return tuple(sorted(values))


def select_annotations(
    annotations: Sequence[Mapping[str, Any]],
    benchmark: str,
    ordinals: tuple[int, ...] | None,
    split: str,
    *,
    seed: int = 260809,
    num_chunks: int = 1,
    chunk_idx: int = 0,
) -> list[tuple[int, Mapping[str, Any]]]:
    if split not in {"all", "dev", "holdout"}:
        raise ValueError("split must be all, dev, or holdout")
    if isinstance(num_chunks, bool) or not isinstance(num_chunks, int) or num_chunks < 1:
        raise ValueError("num_chunks must be at least one")
    if isinstance(chunk_idx, bool) or not isinstance(chunk_idx, int) or not 0 <= chunk_idx < num_chunks:
        raise ValueError("chunk_idx must be within [0, num_chunks)")
    if ordinals is not None:
        if any(isinstance(ordinal, bool) or not isinstance(ordinal, int) for ordinal in ordinals):
            raise TypeError("ordinals must contain only integers")
        if any(ordinal < 0 for ordinal in ordinals):
            raise ValueError("ordinals must be non-negative")
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("ordinals must not contain duplicates")
    indexed = list(enumerate(annotations))
    split_rows = [
        (ordinal, row) for ordinal, row in indexed
        if split == "all" or split_bucket(benchmark, row["input_image"], seed) == split
    ]
    if ordinals is not None:
        requested = set(ordinals)
        out_of_range = [ordinal for ordinal in ordinals if ordinal >= len(indexed)]
        if out_of_range:
            raise ValueError(f"ordinals out of range: {out_of_range}")
        available = {ordinal for ordinal, _ in split_rows}
        excluded = sorted(requested - available)
        if excluded:
            raise ValueError(f"ordinals excluded by split: {excluded}")
        split_rows = [(ordinal, row) for ordinal, row in split_rows if ordinal in requested]
    selected = split_rows[chunk_idx::num_chunks]
    if not selected:
        raise ValueError("annotation selection is empty")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--annotation-path", "--annotation_path", dest="annotation_path", required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--config", required=True, help="minimal_v1 or a strict JSON config path")
    parser.add_argument("--nlp-model-path", default="models/en_core_web_sm-3.8.0")
    parser.add_argument("--sam-model-path", default="models/facebook/sam3/sam3.pt")
    parser.add_argument("--clip-model-path", default=CLIP_SNAPSHOT)
    parser.add_argument("--mode", choices=("rerank_only", "root_search_fallback"))
    parser.add_argument("--ordinals", help="global ordinals, e.g. 1,4-7,20")
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    parser.add_argument("--split-seed", type=int, default=260809)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _annotation_file(annotation_root: Path, benchmark: str) -> tuple[Path, str]:
    if benchmark.startswith("fines-bench"):
        return annotation_root / "fines-bench" / f"annotation_{benchmark}.json", "fines-bench"
    return annotation_root / benchmark / f"annotation_{benchmark}.json", benchmark


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("annotation file must contain a JSON list of objects")
    return rows


def _fingerprint(
    *, benchmark: str, paths: Mapping[str, Path], config: Mapping[str, Any], split: str,
    split_seed: int, ordinals: Sequence[int], num_chunks: int, chunk_idx: int,
) -> str:
    payload = {
        "runner_version": RUNNER_VERSION,
        "benchmark": benchmark,
        "paths": {name: str(path.resolve()) for name, path in sorted(paths.items())},
        "config": dict(config),
        "split": split,
        "split_seed": split_seed,
        "ordinals": list(ordinals),
        "num_chunks": num_chunks,
        "chunk_idx": chunk_idx,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_output(benchmark: str, policy: Mapping[str, Any], output: Any) -> None:
    if benchmark == "vstar":
        if isinstance(output, bool) or not isinstance(output, int):
            raise ValueError("V* output must be a plain integer")
        options = policy["options"]
        if not isinstance(options, list) or not 0 <= output < len(options):
            raise ValueError("V* output is outside the option range")
    elif benchmark in {"hr-bench_4k", "hr-bench_8k"}:
        options = policy["options"]
        if not isinstance(output, list) or not isinstance(options, list) or len(output) != 4 or len(options) != 4:
            raise ValueError("HR-Bench requires exactly four option blocks and outputs")
        if not all(isinstance(item, str) for item in output):
            raise ValueError("HR-Bench outputs must be strings")


def _load_runtime(model_path: Path, sam_path: Path, nlp_path: Path, clip_path: Path,
                  rerank_enabled: bool) -> tuple[Any, Any, Any, Any, Any]:
    import spacy
    import torch

    cvsearch_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(cvsearch_dir))
    from CVSearch import get_cvsearch_response
    from models.modeling_qwenvl import ModelQwenVL
    from models.modeling_sam3 import sam3_inference

    if "qwen" not in str(model_path).casefold():
        raise ValueError("Task 7 supports the fixed Qwen CVSearch model only")
    kwargs = {"load_in_8bit": True} if "32b" in str(model_path).casefold() else {}
    zoom_model = ModelQwenVL(
        model_path=str(model_path), device="cuda:0", torch_dtype=torch.bfloat16,
        patch_scale=1.2, **kwargs,
    )
    sam_model = sam3_inference(model_path=str(sam_path))
    nlp_model = spacy.load(name=str(nlp_path))
    if rerank_enabled:
        from cvsearch.evidence_gap.clip_scorer import ClipScorer

        scorer = ClipScorer(device="cuda", model_path=str(clip_path))
    else:
        scorer = None
    return sam_model, zoom_model, nlp_model, scorer, get_cvsearch_response


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.force:
        raise ValueError("--resume and --force are mutually exclusive")
    ordinals = parse_ordinals(args.ordinals)
    config = load_method_config(args.config)
    if args.mode is not None:
        config["mode"] = args.mode
        config = load_method_config(config)

    root = Path(args.root_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = {
        "model": _resolve(root, args.model_path),
        "annotations": _resolve(root, args.annotation_path),
        "sam": _resolve(root, args.sam_model_path),
        "nlp": _resolve(root, args.nlp_model_path),
        "clip": _resolve(root, args.clip_model_path),
    }
    if not paths["model"].is_dir():
        raise FileNotFoundError(paths["model"])
    if not paths["annotations"].is_dir():
        raise FileNotFoundError(paths["annotations"])
    if not paths["sam"].is_file():
        raise FileNotFoundError(paths["sam"])
    if not paths["nlp"].exists():
        raise FileNotFoundError(paths["nlp"])
    if config["rerank_enabled"] and not paths["clip"].is_dir():
        raise FileNotFoundError(paths["clip"])

    annotation_file, image_benchmark = _annotation_file(paths["annotations"], args.benchmark)
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)
    ic_path = Path(__file__).resolve().parent / "ic_examples" / f"{args.benchmark}.json"
    if not ic_path.is_file():
        raise FileNotFoundError(ic_path)
    rows = _load_rows(annotation_file)
    selected = select_annotations(
        rows, args.benchmark, ordinals, args.split, seed=args.split_seed,
        num_chunks=args.num_chunks, chunk_idx=args.chunk_idx,
    )
    expected = tuple(ordinal for ordinal, _ in selected)

    answers_path = Path(args.answers_file).expanduser().resolve()
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(
        benchmark=args.benchmark, paths=dict(paths, annotation_file=annotation_file, ic_examples=ic_path),
        config=config, split=args.split, split_seed=args.split_seed, ordinals=expected,
        num_chunks=args.num_chunks, chunk_idx=args.chunk_idx,
    )
    writer = JsonlCheckpointWriter(
        answers_path, expected, resume=args.resume, run_fingerprint=fingerprint,
        allow_replace=args.force,
    )
    try:
        missing = [(ordinal, row) for ordinal, row in selected if ordinal not in writer.completed]
        if not missing:
            writer.finalize()
            return 0
        sam_model, zoom_model, nlp_model, scorer, cvsearch_fn = _load_runtime(
            paths["model"], paths["sam"], paths["nlp"], paths["clip"], config["rerank_enabled"]
        )
        with ic_path.open("r", encoding="utf-8") as handle:
            ic_examples = json.load(handle)
        image_folder = paths["annotations"] / image_benchmark
        for ordinal, original in missing:
            policy = sanitize_annotation(original)
            response, trace = get_evidence_gap_response(
                sam_model=sam_model,
                zoom_model=zoom_model,
                nlp_model=nlp_model,
                policy_annotation=policy,
                original_annotation=original,
                ic_examples=ic_examples,
                decomposed_question_template="What is the appearance of the {}?",
                config=config,
                image_folder=str(image_folder),
                cvsearch_fn=cvsearch_fn,
                scorer=scorer,
            )
            _validate_output(args.benchmark, policy, response)
            writer.write(ordinal, compose_output_record(original, response, trace))
        writer.finalize()
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
