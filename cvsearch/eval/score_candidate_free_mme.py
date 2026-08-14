#!/usr/bin/env python3
"""Score the single sealed MME-RealWorld-Lite robust-transfer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.answers import official_letter


BACKBONES = ("qwen", "internvl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_external_gate(
    backbones: Mapping[str, Mapping[str, int]], *, provenance_ok: bool,
) -> dict[str, Any]:
    """Evaluate the predeclared external-only transfer gate."""
    if set(backbones) != set(BACKBONES):
        raise ValueError("external gate requires exactly qwen and internvl")
    failures = []
    for backbone in BACKBONES:
        metrics = backbones[backbone]
        if metrics.get("delta", 0) < 0:
            failures.append(f"{backbone} backbone regressed")
    corrections = sum(backbones[name].get("corrections", 0) for name in BACKBONES)
    corruptions = sum(backbones[name].get("corruptions", 0) for name in BACKBONES)
    pooled_delta = sum(backbones[name].get("delta", 0) for name in BACKBONES)
    if pooled_delta <= 0:
        failures.append("pooled delta is not strictly positive")
    if corrections <= corruptions:
        failures.append("corrections do not exceed corruptions")
    if provenance_ok is not True:
        failures.append("provenance audit failed")
    return {
        "pooled_delta": pooled_delta,
        "corrections": corrections,
        "corruptions": corruptions,
        "provenance_ok": provenance_ok,
        "failures": failures,
        "passed": not failures,
    }


def _read_decisions(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            ordinal = row.get("source_ordinal") if type(row) is dict else None
            if type(ordinal) is not int or ordinal < 0 or ordinal in rows:
                raise ValueError(f"invalid decision ordinal at {path}:{line_number}")
            rows[ordinal] = row
    return rows


def _runtime_manifest(path: Path, backbone: str) -> tuple[dict[str, Any], bool]:
    manifest_path = Path(f"{path}.runtime-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid = bool(
        type(manifest) is dict
        and manifest.get("artifact_kind") == "robust-transfer-external-decisions"
        and manifest.get("data_scope") == "mme-realworld-lite-label-blind"
        and manifest.get("backbone") == backbone
        and manifest.get("output_sha256") == _sha256(path)
    )
    return manifest, valid


def _score_backbone(
    annotations: Sequence[Mapping[str, Any]],
    decisions: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if len(decisions) != len(annotations) or set(decisions) != set(range(len(annotations))):
        raise ValueError("external decisions do not cover the annotation ordinals exactly")
    baseline_correct = robust_correct = corrections = corruptions = 0
    observations = selections = failures = 0
    for ordinal, annotation in enumerate(annotations):
        decision = decisions[ordinal]
        if decision.get("input_image") != annotation.get("input_image"):
            raise ValueError(f"external image identity mismatch at ordinal {ordinal}")
        truth = annotation.get("Ground truth")
        if not isinstance(truth, str) or truth not in "ABCDE":
            raise ValueError(f"invalid MME ground truth at ordinal {ordinal}")
        p0 = official_letter(decision.get("p0_output"), allowed="ABCDE")
        selected = official_letter(
            decision.get("selected_output"), allowed="ABCDE",
        )
        if p0 is None or selected is None:
            raise ValueError(f"unparseable external output at ordinal {ordinal}")
        before = p0 == truth
        after = selected == truth
        baseline_correct += before
        robust_correct += after
        corrections += not before and after
        corruptions += before and not after
        observations += decision.get("observations", 0)
        selections += decision.get("selected_source") != "P0"
        failures += decision.get("failure") is not None
    return {
        "topics": len(annotations),
        "baseline_correct": baseline_correct,
        "robust_correct": robust_correct,
        "delta": robust_correct - baseline_correct,
        "corrections": corrections,
        "corruptions": corruptions,
        "selections": selections,
        "runtime_failures": failures,
        "observations": observations,
        "mean_observations": observations / len(annotations),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = json.loads(args.annotation.read_text(encoding="utf-8"))
    if not isinstance(annotations, list) or not all(
        isinstance(row, Mapping) for row in annotations
    ):
        raise ValueError("MME annotation must be a list of objects")
    decision_paths = {"qwen": args.qwen, "internvl": args.internvl}
    manifests = {}
    manifest_validity = {}
    metrics = {}
    for backbone in BACKBONES:
        path = decision_paths[backbone]
        manifests[backbone], manifest_validity[backbone] = _runtime_manifest(
            path, backbone,
        )
        decisions = _read_decisions(path)
        metrics[backbone] = _score_backbone(annotations, decisions)
        manifest_validity[backbone] &= (
            manifests[backbone].get("records") == len(decisions)
            and manifests[backbone].get("policy_sha256")
            == manifests[BACKBONES[0]].get("policy_sha256")
        )
    gate = score_external_gate(
        metrics, provenance_ok=all(manifest_validity.values()),
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "robust-transfer-mme-realworld-lite-one-shot",
        "evaluation_count": 1,
        "annotation_path": str(args.annotation.resolve()),
        "annotation_sha256": _sha256(args.annotation),
        "backbones": metrics,
        "runtime_manifests": {
            backbone: {
                "path": str(Path(f"{decision_paths[backbone]}.runtime-manifest.json")),
                "sha256": _sha256(Path(f"{decision_paths[backbone]}.runtime-manifest.json")),
                "valid": manifest_validity[backbone],
            }
            for backbone in BACKBONES
        },
        "gate": gate,
        "passed": gate["passed"],
    }
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--internvl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report["gate"], sort_keys=True), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
