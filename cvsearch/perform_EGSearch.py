#!/usr/bin/env python3
"""Resumable, sanitized runner for the minimal evidence-gap method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvsearch.evidence_gap.clip_scorer import CLIP_SNAPSHOT
from cvsearch.evidence_gap.answers import official_letter, single_choice_allowed
from cvsearch.evidence_gap.input import sanitize_annotation, split_bucket
from cvsearch.evidence_gap.io import JsonlCheckpointWriter
from cvsearch.evidence_gap.method import compose_output_record, get_evidence_gap_response, load_method_config
from cvsearch.evidence_gap.provenance import (
    build_launch_manifest,
    canonical_sha256,
    runtime_environment,
    visible_gpu_uuids,
    write_or_validate_manifest,
)


BENCHMARKS = (
    "vstar", "hr-bench_4k", "hr-bench_8k", "mme-realworld-lite", "treebench",
    "fines-bench_option", "fines-bench_reasoning",
)
RUNNER_VERSION = "evidence-gap-phase2-v2"
_MME_OPTION_LINE = re.compile(r"^\s*(?:\(([A-E])\)|([A-E])\.)\s*(\S.*?)\s*$")
_CODE_REVISION_EXACT_PATHS = (
    Path("cvsearch/perform_EGSearch.py"),
    Path("cvsearch/CVSearch.py"),
    Path("cvsearch/models/modeling_qwenvl.py"),
    Path("cvsearch/models/modeling_sam3.py"),
    Path("cvsearch/models/tree.py"),
    Path("cvsearch/models/utils.py"),
    Path("cvsearch/eval/phase2_oracle.py"),
)
_CODE_REVISION_TREES = (
    Path("cvsearch/evidence_gap"),
    Path("cvsearch/models"),
    Path("llava"),
    Path("sam3"),
)


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
    lexical = path if path.is_absolute() else root / path
    if lexical.is_symlink():
        raise ValueError(f"artifact path may not be a symlink: {lexical}")
    return lexical.resolve()


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


def _code_manifest(source_root: Path | None = None) -> list[dict[str, str]]:
    root = Path(__file__).resolve().parents[1] if source_root is None else Path(source_root).resolve()
    relative_paths = set(_CODE_REVISION_EXACT_PATHS)
    for relative_tree in _CODE_REVISION_TREES:
        tree = root / relative_tree
        if not tree.is_dir():
            raise FileNotFoundError(tree)
        relative_paths.update(path.relative_to(root) for path in tree.rglob("*.py"))
    manifest = []
    for relative in sorted(relative_paths, key=lambda path: path.as_posix()):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest.append({
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return manifest


def _code_revision(source_root: Path | None = None) -> str:
    manifest = _code_manifest(source_root)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint(
    *, benchmark: str, paths: Mapping[str, Path], config: Mapping[str, Any], split: str,
    split_seed: int, ordinals: Sequence[int], num_chunks: int, chunk_idx: int,
    code_revision: str,
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
        "code_revision": code_revision,
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
    elif benchmark == "treebench":
        if not isinstance(output, str):
            raise ValueError("TreeBench output must be a string")
    elif benchmark == "mme-realworld-lite":
        allowed = single_choice_allowed(policy.get("options"))
        if allowed != "ABCDE":
            raise ValueError("MME policy must expose exactly five contiguous A-E options")
        if not isinstance(output, str) or official_letter(output, allowed=allowed) is None:
            raise ValueError("MME output must contain a canonical A-E option letter")


def _normalize_policy(benchmark: str, annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert MME's visible list schema to the existing option-single contract."""
    policy = sanitize_annotation(annotation)
    if benchmark != "mme-realworld-lite":
        return policy
    if policy["answer_type"] != "Multiple Choice":
        raise ValueError("MME annotation must use the Multiple Choice schema")
    options = policy["options"]
    if (
        not isinstance(options, list)
        or len(options) != 5
        or not all(isinstance(option, str) and option for option in options)
    ):
        raise ValueError("MME requires exactly five nonempty option strings")
    normalized_options = []
    labels = []
    for option in options:
        match = _MME_OPTION_LINE.fullmatch(option)
        if match is None:
            raise ValueError("MME options must be exact contiguous A-E lines")
        label = match.group(1) or match.group(2)
        labels.append(label)
        normalized_options.append(f"{label}. {match.group(3)}")
    option_block = "\n".join(normalized_options)
    if labels != list("ABCDE") or single_choice_allowed(option_block) != "ABCDE":
        raise ValueError("MME options must be exact contiguous A-E lines")
    policy["answer_type"] = "option_single"
    policy["options"] = option_block
    return policy


def _compose_policy_output(
    original: Mapping[str, Any], policy: Mapping[str, Any],
    response: Any, trace: Any,
) -> dict[str, Any]:
    """Carry the exact visible inference schema into downstream replay rows."""
    record = compose_output_record(original, response, trace)
    if (
        original.get("answer_type") == "Multiple Choice"
        and policy.get("answer_type") == "option_single"
    ):
        record["answer_type"] = "option_single"
        record["options"] = policy["options"]
    return record


def _model_family_from_config(model_path: Path) -> str:
    config_path = Path(model_path) / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model config: {config_path}") from error
    if type(config) is not dict:
        raise ValueError("model config must be an exact JSON object")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("model config requires a nonempty model_type")
    normalized = model_type.casefold()
    if normalized in {"qwen2_5_vl", "qwen3_vl"}:
        return "qwen"
    if normalized == "internvl_chat":
        return "internvl"
    if normalized == "llava":
        return "llava"
    raise ValueError(f"unsupported model_type: {model_type!r}")


def _load_runtime(model_path: Path, sam_path: Path, nlp_path: Path, clip_path: Path,
                  rerank_enabled: bool) -> tuple[Any, Any, Any, Any, Any]:
    import spacy
    import torch

    cvsearch_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(cvsearch_dir))
    from CVSearch import get_cvsearch_response
    from models.modeling_sam3 import sam3_inference

    family = _model_family_from_config(model_path)
    if family == "qwen":
        from models.modeling_qwenvl import ModelQwenVL

        kwargs = (
            {"load_in_8bit": True}
            if "32b" in str(model_path).casefold() else {}
        )
        zoom_model = ModelQwenVL(
            model_path=str(model_path), device="cuda:0",
            torch_dtype=torch.bfloat16, patch_scale=1.2, **kwargs,
        )
    elif family == "internvl":
        from models.modeling_internvl import ModelInternvl

        zoom_model = ModelInternvl(
            model_path=str(model_path), device="cuda:0",
            torch_dtype=torch.bfloat16, patch_scale=1.2,
        )
    else:
        from models.modeling_llava import ModelGlobalLocal, ModelLocal

        with (model_path / "config.json").open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        if "anyres" in str(model_config.get("image_aspect_ratio", "")).casefold():
            zoom_model = ModelGlobalLocal(
                model_path=str(model_path), conv_type="qwen_1_5",
                device="cuda:0", torch_dtype=torch.bfloat16,
                patch_scale=1.2, bias_value=0.6,
            )
        else:
            zoom_model = ModelLocal(
                model_path=str(model_path), conv_type="v1", device="cuda:0",
                torch_dtype=torch.bfloat16, patch_scale=None, bias_value=0.2,
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
    config_source = Path(args.config).expanduser()
    if config_source.is_symlink():
        raise ValueError("config path may not be a symlink")
    if not config_source.is_file():
        raise ValueError("reproducible runs require a strict JSON config file")
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
    if annotation_file.is_symlink():
        raise ValueError("annotation file may not be a symlink")
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)
    ic_path = Path(__file__).resolve().parent / "ic_examples" / f"{args.benchmark}.json"
    if ic_path.is_symlink():
        raise ValueError("in-context examples file may not be a symlink")
    if not ic_path.is_file():
        raise FileNotFoundError(ic_path)
    rows = _load_rows(annotation_file)
    selected = select_annotations(
        rows, args.benchmark, ordinals, args.split, seed=args.split_seed,
        num_chunks=args.num_chunks, chunk_idx=args.chunk_idx,
    )
    expected = tuple(ordinal for ordinal, _ in selected)

    image_folder = paths["annotations"] / image_benchmark
    source_images = []
    seen_images = set()
    for _, row in selected:
        image_value = row.get("input_image")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("selected annotation input_image must be a nonempty string")
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = image_folder / image_path
        lexical_key = str(image_path.absolute())
        if lexical_key not in seen_images:
            seen_images.add(lexical_key)
            source_images.append(image_path)

    answers_path = Path(args.answers_file).expanduser().resolve()
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    code_manifest = _code_manifest()
    code_revision = hashlib.sha256(json.dumps(
        code_manifest, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    launch_manifest = build_launch_manifest(
        benchmark=args.benchmark, code_revision=code_revision,
        code_manifest={"files": code_manifest}, loaded_config=config,
        config_source=config_source, annotation_file=annotation_file,
        selected_rows=selected, source_images=source_images, ic_examples=ic_path,
        model_path=paths["model"], sam_path=paths["sam"], spacy_path=paths["nlp"],
        clip_path=paths["clip"] if config["rerank_enabled"] else None,
        split=args.split, split_seed=args.split_seed,
        num_chunks=args.num_chunks, chunk_idx=args.chunk_idx,
        environment=runtime_environment(), gpu_uuids=visible_gpu_uuids(),
    )
    fingerprint = canonical_sha256(launch_manifest)
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
        sam_model, zoom_model, nlp_model, scorer, cvsearch_fn = _load_runtime(
            paths["model"], paths["sam"], paths["nlp"], paths["clip"], config["rerank_enabled"]
        )
        with ic_path.open("r", encoding="utf-8") as handle:
            ic_examples = json.load(handle)
        for ordinal, original in missing:
            policy = _normalize_policy(args.benchmark, original)
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
            writer.write(
                ordinal,
                _compose_policy_output(original, policy, response, trace),
            )
        writer.finalize()
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
