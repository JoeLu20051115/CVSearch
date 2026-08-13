#!/usr/bin/env python3
"""Collect label-blind order-symmetric verifier losses on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from cvsearch.eval.analyze_split_search import load_selected_calibrations
from cvsearch.eval.freeze_uncertainty_support import BENCHMARKS
from cvsearch.eval.pairwise_verifier import (
    CONFIDENCE_THRESHOLDS,
    PROPOSAL_AGREEMENTS,
    PairwiseProjection,
    PairwiseProposal,
    compose_independent_source_view,
    compose_pairwise_evidence_sheet,
    pairwise_answer_display,
    pairwise_decision,
    pairwise_prompt_material,
    project_pairwise_losses,
    propose_pairwise_candidate,
    select_pairwise_evidence_views,
    source_pairwise_prompt_material,
)
from cvsearch.eval.replay_adaptive_search import replay_adaptive_search
from cvsearch.eval.replay_uncertainty_support import sanitize_replay_row
from cvsearch.eval.run_split_search import _load_model, _model_family
from cvsearch.evidence_gap.provenance import canonical_sha256


PAIRWISE_PROCESSOR_MODES = {
    "candidate_crops": "marked_overview_two_agreeing_crops_order_reversed",
    "independent_source": "complete_source_short_choice_order_reversed",
}


def _failure(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }


def _fallback_proposal(
    stage2_row: Mapping[str, Any], calibration: Any,
) -> PairwiseProposal:
    output = stage2_row.get("output")
    try:
        sanitized = sanitize_replay_row(stage2_row)
        output = replay_adaptive_search(
            sanitized, sanitized, calibration,
        )["selected_output"]
    except (KeyError, TypeError, ValueError):
        pass
    return PairwiseProposal(
        feasible=False,
        stage2_output=output,
        candidate_output=None,
        p0_canonical=None,
        candidate_canonical=None,
        observations=0,
        agreement=0.0,
        agreeing_hashes=(),
    )


def _decision_grid(
    proposal: PairwiseProposal,
    projection: PairwiseProjection | None,
) -> dict[str, dict[str, Any]]:
    return {
        f"agreement={agreement:g},confidence={confidence:g}": pairwise_decision(
            proposal,
            projection,
            agreement_threshold=agreement,
            confidence_threshold=confidence,
        )
        for agreement in PROPOSAL_AGREEMENTS
        for confidence in CONFIDENCE_THRESHOLDS
    }


def produce_pairwise_record(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    calibration: Any,
    source: Image.Image,
    model: Any,
    *,
    evidence_mode: str = "candidate_crops",
) -> dict[str, Any]:
    """Produce one label-blind verifier record and fail closed to exact P0."""
    if evidence_mode not in PAIRWISE_PROCESSOR_MODES:
        raise ValueError("pairwise evidence mode is unsupported")
    failure = None
    proposal = _fallback_proposal(stage2_row, calibration)
    projection = None
    render_audit = None
    prompt_audit = None
    observations = None
    charged_calls = 0
    try:
        proposal = propose_pairwise_candidate(
            stage2_row, split_row, calibration,
            minimum_agreement=min(PROPOSAL_AGREEMENTS),
        )
        if proposal.feasible:
            p0_display = pairwise_answer_display(
                stage2_row["options"], proposal.p0_canonical,
            )
            candidate_display = pairwise_answer_display(
                stage2_row["options"], proposal.candidate_canonical,
            )
            if evidence_mode == "candidate_crops":
                views = select_pairwise_evidence_views(split_row, proposal)
                sheet, render_audit = compose_pairwise_evidence_sheet(source, views)
                material = pairwise_prompt_material(
                    stage2_row["question"], stage2_row["options"],
                    p0_display, candidate_display,
                )
            else:
                sheet, render_audit = compose_independent_source_view(source)
                material = source_pairwise_prompt_material(
                    stage2_row["question"], p0_display, candidate_display,
                )
            prompt_audit = {
                "prompt_sha256": material["prompt_sha256"],
                "choices_sha256": canonical_sha256(material["choices"]),
                "candidate_choice_indices": material["candidate_choice_indices"],
            }
            observations = []
            for prompt in material["prompts"]:
                charged_calls += 1
                winner, losses = model.multiple_choices_with_losses(
                    sheet.copy(), prompt, list(material["choices"]), [],
                )
                observations.append({
                    "winner": int(winner),
                    "losses": [float(value) for value in losses],
                })
            projection = project_pairwise_losses(observations)
    except Exception as error:
        failure = _failure(error)
        projection = None

    decisions = _decision_grid(proposal, projection)
    planned_calls = 2 if proposal.feasible else 0
    return {
        "evidence_mode": evidence_mode,
        "proposal": asdict(proposal),
        "render_audit": render_audit,
        "prompt_audit": prompt_audit,
        "observations": observations,
        "projection": None if projection is None else asdict(projection),
        "decisions": decisions,
        "failure": failure,
        "cost": {
            "proposal_observations": proposal.observations,
            "planned_verifier_calls": planned_calls,
            "charged_verifier_calls": charged_calls,
            "observations": proposal.observations + planned_calls,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"pairwise input is not a file: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank pairwise input row: {path}:{line_number}")
        row = json.loads(line)
        if type(row) is not dict:
            raise ValueError(f"pairwise input row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _indexed(rows: Sequence[Mapping[str, Any]], name: str) -> dict[int, Mapping[str, Any]]:
    result = {}
    for row in rows:
        ordinal = row.get("_eg_ordinal")
        if type(ordinal) is not int or ordinal < 0 or ordinal in result:
            raise ValueError(f"{name} has an invalid or duplicate ordinal")
        result[ordinal] = row
    return result


def _split_manifest(path: Path, backbone: str) -> dict[str, Any]:
    manifest_path = Path(f"{path}.split-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        type(manifest) is not dict
        or manifest.get("model_family") != backbone
        or manifest.get("output_sha256") != _sha256_file(path)
        or manifest.get("records") != len(_read_jsonl(path))
        or not isinstance(manifest.get("model_path"), str)
        or not isinstance(manifest.get("image_root"), str)
    ):
        raise ValueError(f"split manifest does not bind its input: {path}")
    return manifest


def _resolve_source(
    row: Mapping[str, Any], image_root: Path,
) -> Image.Image:
    value = row.get("input_image")
    if not isinstance(value, str) or not value:
        raise ValueError("pairwise input image must be nonempty")
    root = image_root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("pairwise input image escapes its frozen root")
    with Image.open(path) as image:
        return image.convert("RGB")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = Path(f"{path}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _partition_rows(
    development_root: Path,
    stage2_root: Path,
    split_root: Path,
    backbone: str,
) -> tuple[list[tuple[str, str, Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]], list[dict[str, Any]]]:
    items = []
    bindings = []
    for partition in ("development", "validation_v3"):
        for benchmark in sorted(BENCHMARKS):
            split_path = (
                development_root if partition == "development" else split_root
            ) / backbone / f"{benchmark}.jsonl"
            stage2_path = (
                split_path
                if partition == "development"
                else stage2_root / backbone / f"{benchmark}.jsonl"
            )
            split_rows = _indexed(_read_jsonl(split_path), f"{partition} SPLIT")
            stage2_rows = _indexed(_read_jsonl(stage2_path), f"{partition} Stage-2")
            if split_rows.keys() != stage2_rows.keys():
                raise ValueError(f"{partition}/{backbone}/{benchmark} ordinals differ")
            manifest = _split_manifest(split_path, backbone)
            bindings.append({
                "partition": partition,
                "backbone": backbone,
                "benchmark": benchmark,
                "stage2_path": str(stage2_path),
                "stage2_sha256": _sha256_file(stage2_path),
                "split_path": str(split_path),
                "split_sha256": _sha256_file(split_path),
                "split_manifest_sha256": _sha256_file(
                    Path(f"{split_path}.split-manifest.json")
                ),
            })
            for ordinal in sorted(split_rows):
                items.append((
                    partition, benchmark, split_path,
                    stage2_rows[ordinal], split_rows[ordinal], manifest,
                ))
    return items, bindings


def _record_identity(
    partition: str, backbone: str, benchmark: str,
    stage2_row: Mapping[str, Any],
    evidence_mode: str,
    verifier_model_path: Path,
) -> str:
    return canonical_sha256({
        "partition": partition,
        "backbone": backbone,
        "benchmark": benchmark,
        "evidence_mode": evidence_mode,
        "verifier_model_path": str(verifier_model_path),
        "source_ordinal": stage2_row.get("_eg_ordinal"),
        "input_image": stage2_row.get("input_image"),
        "question": stage2_row.get("question"),
        "options": stage2_row.get("options"),
    })


def _load_partial(path: Path, expected: Sequence[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_jsonl(path)
    identities = [row.get("input_identity_sha256") for row in rows]
    if identities != list(expected[:len(rows)]):
        raise ValueError("partial pairwise output is not an exact input prefix")
    return rows


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    calibrations = load_selected_calibrations(args.support_calibration)
    calibration = calibrations.get(args.backbone)
    if calibration is None:
        raise ValueError(f"support calibration is missing for {args.backbone}")
    items, bindings = _partition_rows(
        args.development_root, args.stage2_root, args.split_root, args.backbone,
    )
    model_paths = {item[-1]["model_path"] for item in items}
    if len(model_paths) != 1:
        raise ValueError("one backbone collection must bind one model path")
    source_model_path = Path(next(iter(model_paths)))
    if _model_family(source_model_path) != args.backbone:
        raise ValueError("model family differs from requested backbone")
    verifier_model_path = (
        source_model_path
        if args.verifier_model_path is None
        else args.verifier_model_path
    )
    _model_family(verifier_model_path)
    output = args.output
    manifest_path = Path(f"{output}.pairwise-manifest.json")
    if output.exists() or output.is_symlink() or manifest_path.exists():
        raise FileExistsError(output)
    partial = Path(f"{output}.partial")
    identities = [
        _record_identity(
            partition, args.backbone, benchmark, stage2, args.evidence_mode,
            verifier_model_path,
        )
        for partition, benchmark, _, stage2, _, _ in items
    ]
    records = _load_partial(partial, identities)
    start = len(records)
    model = _load_model(verifier_model_path) if start < len(items) else None
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if partial.exists() else "x"
    with partial.open(mode, encoding="utf-8") as stream:
        for index in range(start, len(items)):
            partition, benchmark, _, stage2, split, split_manifest = items[index]
            if stage2.get("input_image") != split.get("input_image"):
                raise ValueError("Stage-2 and SPLIT input images drifted")
            image_root = Path(split_manifest["image_root"])
            if not image_root.is_absolute():
                image_root = args.workspace_root / image_root
            source = _resolve_source(stage2, image_root)
            record = produce_pairwise_record(
                stage2, split, calibration, source, model,
                evidence_mode=args.evidence_mode,
            )
            record.update({
                "partition": partition,
                "backbone": args.backbone,
                "benchmark": benchmark,
                "source_ordinal": stage2["_eg_ordinal"],
                "input_identity_sha256": identities[index],
            })
            stream.write(json.dumps(
                record, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            print(
                f"[{index + 1}/{len(items)}] {partition}/{args.backbone}/"
                f"{benchmark}/{stage2['_eg_ordinal']} "
                f"proposal={record['proposal']['feasible']} "
                f"verifier={record['projection'] is not None}",
                file=sys.stderr, flush=True,
            )
    os.replace(partial, output)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "pairwise-uncertainty-verifier-observations",
        "data_scope": "opened_development_label_blind",
        "processor_mode": PAIRWISE_PROCESSOR_MODES[args.evidence_mode],
        "evidence_mode": args.evidence_mode,
        "backbone": args.backbone,
        "source_model_path": str(source_model_path),
        "source_model_config_sha256": _sha256_file(
            source_model_path / "config.json"
        ),
        "verifier_model_path": str(verifier_model_path),
        "verifier_model_config_sha256": _sha256_file(
            verifier_model_path / "config.json"
        ),
        "shared_external_verifier": verifier_model_path != source_model_path,
        "support_calibration_sha256": calibration.manifest_sha256,
        "input_bindings": bindings,
        "input_bindings_sha256": canonical_sha256(bindings),
        "records": len(records),
        "feasible_proposals": sum(row["proposal"]["feasible"] for row in records),
        "charged_verifier_calls": sum(
            row["cost"]["charged_verifier_calls"] for row in records
        ),
        "output_sha256": _sha256_file(output),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--support-calibration", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--backbone", choices=("qwen", "internvl"), required=True)
    parser.add_argument("--verifier-model-path", type=Path)
    parser.add_argument(
        "--evidence-mode", choices=tuple(PAIRWISE_PROCESSOR_MODES),
        default="candidate_crops",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run_collection(build_parser().parse_args(argv))
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
