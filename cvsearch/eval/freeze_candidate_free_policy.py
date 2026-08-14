#!/usr/bin/env python3
"""Freeze the accepted candidate-free robust-transfer policy before MME scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cvsearch.eval.analyze_split_search import (
    load_selected_calibrations,
    official_correctness,
)
from cvsearch.eval.candidate_free_verifier_selector import (
    CandidateFreeVerifierEvidence,
)
from cvsearch.eval.freeze_uncertainty_support import (
    DevelopmentRecord,
    _risk_topics,
    source_group,
)
from cvsearch.eval.replay_uncertainty_support import _prepare_replay
from cvsearch.eval.robust_transfer_selector import (
    AcceptanceCriteria,
    _metrics_for_topics,
    evaluate_acceptance,
    select_shared_configuration,
    verifier_evidence_key,
)


DEFAULT_ROOT = Path("reproduction/evidence_gap/adaptive_search_v18")
DEFAULT_OBSERVATIONS = (
    DEFAULT_ROOT / "independent-answer-qwen25vl32b-4mp-qwen.jsonl",
    DEFAULT_ROOT / "independent-answer-qwen25vl32b-4mp-internvl.jsonl",
    DEFAULT_ROOT / "independent-answer-qwen25vl32b-4mp-historical-qwen.jsonl",
    DEFAULT_ROOT / "independent-answer-qwen25vl32b-4mp-historical-internvl-v2.jsonl",
)
DEFAULT_CALIBRATION = Path(
    "reproduction/evidence_gap/adaptive_search_v7/"
    "split-calibration-manifest-v2.json"
)
DEFAULT_ACCEPTED_REPORT = DEFAULT_ROOT / "qwen25vl32b-current-development-report.json"
FREEZE_CODE_PATHS = (
    Path("cvsearch/perform_EGSearch.py"),
    Path("cvsearch/evidence_gap/answers.py"),
    Path("cvsearch/evidence_gap/method.py"),
    Path("cvsearch/evidence_gap/types.py"),
    Path("cvsearch/eval/apply_candidate_free_policy.py"),
    Path("cvsearch/eval/freeze_candidate_free_policy.py"),
    Path("cvsearch/eval/robust_transfer_selector.py"),
    Path("cvsearch/eval/candidate_free_verifier_selector.py"),
    Path("cvsearch/eval/freeze_uncertainty_support.py"),
    Path("cvsearch/eval/pairwise_verifier.py"),
    Path("cvsearch/eval/pairwise_verifier_runner.py"),
    Path("cvsearch/eval/replay_uncertainty_support.py"),
    Path("cvsearch/eval/run_split_search.py"),
    Path("cvsearch/eval/score_candidate_free_mme.py"),
)


def _sha256(path: Path) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if type(value) is not dict:
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if type(row) is not dict:
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _indexed(path: Path) -> dict[int, dict[str, Any]]:
    indexed = {}
    for row in _read_jsonl(path):
        ordinal = row.get("_eg_ordinal")
        if type(ordinal) is not int or ordinal < 0 or ordinal in indexed:
            raise ValueError(f"invalid or duplicate _eg_ordinal in {path}")
        indexed[ordinal] = row
    return indexed


def verified_pairwise_manifest(observation_path: Path) -> dict[str, Any]:
    """Authenticate one verifier JSONL against its exact sidecar manifest."""
    observation_path = Path(observation_path)
    manifest_path = Path(f"{observation_path}.pairwise-manifest.json")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind")
        != "pairwise-uncertainty-verifier-observations"
        or type(manifest.get("input_bindings")) is not list
    ):
        raise ValueError(f"invalid pairwise verifier manifest: {manifest_path}")
    rows = _read_jsonl(observation_path)
    if manifest.get("records") != len(rows):
        raise ValueError(f"pairwise verifier row count mismatch: {observation_path}")
    if manifest.get("output_sha256") != _sha256(observation_path):
        raise ValueError(f"pairwise verifier hash mismatch: {observation_path}")
    return manifest


def _verified_bindings(
    observation_paths: Sequence[Path],
) -> tuple[dict[tuple[str, str, str], tuple[Path, Path]], list[dict[str, Any]]]:
    bindings: dict[tuple[str, str, str], tuple[Path, Path]] = {}
    audits = []
    for observation_path in observation_paths:
        observation_path = Path(observation_path).resolve()
        manifest_path = Path(f"{observation_path}.pairwise-manifest.json")
        manifest = verified_pairwise_manifest(observation_path)
        audits.append({
            "path": str(observation_path),
            "sha256": _sha256(observation_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "records": manifest["records"],
        })
        for item in manifest["input_bindings"]:
            if type(item) is not dict:
                raise ValueError("pairwise input binding must be an object")
            key = (item.get("partition"), item.get("backbone"), item.get("benchmark"))
            if not all(isinstance(value, str) and value for value in key):
                raise ValueError("pairwise input binding identity is invalid")
            stage2_path = Path(item.get("stage2_path", "")).resolve()
            split_path = Path(item.get("split_path", "")).resolve()
            if _sha256(stage2_path) != item.get("stage2_sha256"):
                raise ValueError(f"Stage-2 binding hash mismatch: {key}")
            if _sha256(split_path) != item.get("split_sha256"):
                raise ValueError(f"SPLIT binding hash mismatch: {key}")
            value = (stage2_path, split_path)
            if key in bindings and bindings[key] != value:
                raise ValueError(f"conflicting pairwise input binding: {key}")
            bindings[key] = value
    return bindings, audits


def load_candidate_free_development(
    observation_paths: Sequence[Path], calibration_path: Path,
) -> tuple[dict[str, tuple[DevelopmentRecord, ...]], dict[Any, CandidateFreeVerifierEvidence], list[dict[str, Any]]]:
    """Rebuild the five opened partitions and label-blind verifier evidence."""
    bindings, audits = _verified_bindings(observation_paths)
    calibrations = load_selected_calibrations(Path(calibration_path))
    partitions: dict[str, list[DevelopmentRecord]] = defaultdict(list)
    for (partition, backbone, benchmark), (stage2_path, split_path) in sorted(bindings.items()):
        stage2 = _indexed(stage2_path)
        split = _indexed(split_path)
        if stage2.keys() != split.keys():
            raise ValueError(
                f"Stage-2/SPLIT ordinal mismatch: {(partition, backbone, benchmark)}"
            )
        for ordinal in sorted(stage2):
            partitions[partition].append(DevelopmentRecord(
                group=(
                    f"{partition}:"
                    f"{source_group(benchmark, ordinal, stage2[ordinal].get('input_image'))}"
                ),
                backbone=backbone,
                benchmark=benchmark,
                ordinal=ordinal,
                stage2_row=stage2[ordinal],
                split_row=split[ordinal],
                calibration=calibrations[backbone],
            ))
    frozen_partitions = {
        name: tuple(records) for name, records in sorted(partitions.items())
    }
    records = {
        (partition, record.backbone, record.benchmark, record.ordinal): record
        for partition, values in frozen_partitions.items()
        for record in values
    }
    evidence = {}
    observed_identities = set()
    for observation_path in observation_paths:
        for row in _read_jsonl(Path(observation_path)):
            identity = (
                row.get("partition"), row.get("backbone"),
                row.get("benchmark"), row.get("source_ordinal"),
            )
            if identity not in records or identity in observed_identities:
                raise ValueError(f"unbound or duplicate verifier observation: {identity}")
            observed_identities.add(identity)
            record = records[identity]
            proposal = row.get("proposal")
            projection = row.get("projection")
            if type(proposal) is not dict or projection is not None and type(projection) is not dict:
                raise ValueError(f"invalid verifier observation schema: {identity}")
            corrections = corruptions = 0
            if proposal.get("feasible") is True:
                prepared = _prepare_replay(
                    record.stage2_row, record.split_row, record.calibration,
                )
                before = official_correctness(
                    record.benchmark, record.stage2_row,
                    prepared.stage2["selected_output"],
                )
                after = official_correctness(
                    record.benchmark, record.stage2_row,
                    proposal.get("candidate_output"),
                )
                corrections = sum(not old and new for old, new in zip(before, after))
                corruptions = sum(old and not new for old, new in zip(before, after))
            key = verifier_evidence_key(record)
            evidence[key] = CandidateFreeVerifierEvidence(
                verifier_feasible=bool(projection and projection.get("feasible")),
                verifier_canonical=(
                    projection.get("canonical_answer") if projection else None
                ),
                verifier_confidence=(
                    float(projection.get("confidence", 0.0)) if projection else 0.0
                ),
                proposal_feasible=proposal.get("feasible") is True,
                proposal_canonical=proposal.get("candidate_canonical"),
                proposal_agreement=proposal.get("agreement", 0.0),
                proposal_corrections=corrections,
                proposal_corruptions=corruptions,
                observations=row.get("cost", {}).get("observations"),
            )
    expected_keys = {
        verifier_evidence_key(record)
        for values in frozen_partitions.values()
        for record in values
    }
    if set(evidence) != expected_keys:
        raise ValueError("verifier evidence does not cover every development record")
    return frozen_partitions, evidence, audits


def freeze_policy(
    observation_paths: Sequence[Path] = DEFAULT_OBSERVATIONS,
    calibration_path: Path = DEFAULT_CALIBRATION,
    accepted_report_path: Path = DEFAULT_ACCEPTED_REPORT,
) -> dict[str, Any]:
    """Fit one global refit after the user's explicit +16 acceptance revision."""
    partitions, evidence, observation_audits = load_candidate_free_development(
        observation_paths, calibration_path,
    )
    records = tuple(
        record
        for partition in sorted(partitions)
        for record in partitions[partition]
    )
    criteria = AcceptanceCriteria(minimum_net_gain=16)
    selection = select_shared_configuration(
        records, criteria, verifier_evidence=evidence,
    )
    if selection.failures:
        raise ValueError(f"refit selection failed revised gates: {selection.failures}")
    accepted_report = _read_json(Path(accepted_report_path))
    checkpoint = accepted_report.get("nested_aggregate_then_verifier_cascade")
    if type(checkpoint) is not dict or checkpoint.get("net_gain") != 16:
        raise ValueError("accepted +16 outer checkpoint is missing")
    code_root = Path(__file__).resolve().parents[2]
    code_manifest = [
        {
            "path": relative.as_posix(),
            "sha256": _sha256(code_root / relative),
        }
        for relative in FREEZE_CODE_PATHS
    ]
    selection_payload = selection.to_dict()
    refit_calibrator = dict(selection.refit_calibrators)["*/*"]
    refit_metrics = _metrics_for_topics(
        _risk_topics(records), lambda _: refit_calibrator, evidence,
        proposal_verifier_confidence=(
            selection.configuration.proposal_verifier_confidence
        ),
        verifier_proposal_agreement=(
            selection.configuration.verifier_proposal_agreement
        ),
    )
    refit_failures = evaluate_acceptance(refit_metrics, len(records), criteria)
    if refit_failures:
        raise ValueError(f"deployed refit failed revised gates: {refit_failures}")
    policy_payload = {
        "configuration": selection_payload["configuration"],
        "refit_calibrators": selection_payload["refit_calibrators"],
        "cascade_order": ["AGGREGATE", "VERIFIED_PROPOSAL", "P0"],
        "aggregate_veto_confidence": 0.6,
        "proposal_verifier_confidence": selection.configuration.proposal_verifier_confidence,
        "proposal_agreement": selection.configuration.verifier_proposal_agreement,
    }
    partition_counts = Counter(
        name for name, values in partitions.items() for _ in values
    )
    return {
        "schema_version": 1,
        "artifact_kind": "robust-transfer-candidate-free-policy",
        "status": "frozen_before_mme_outcomes",
        "acceptance_revision": {
            "date": "2026-08-14",
            "accepted_absolute_development_net_gain": 16,
            "superseded_requirement": "scaled +20/512 development recall gate",
            "unchanged_external_gate": (
                "each backbone delta >= 0; pooled delta > 0; "
                "corrections > corruptions"
            ),
        },
        "accepted_outer_checkpoint": {
            "scope": "two opened current partitions nested OOF",
            "report_path": str(Path(accepted_report_path)),
            "report_sha256": _sha256(Path(accepted_report_path)),
            "metrics": checkpoint,
        },
        "refit_scope": {
            "claim": "source-group OOF final refit; not new outer-partition evidence",
            "partitions": dict(sorted(partition_counts.items())),
            "records": len(records),
            "official_units": sum(
                len(official_correctness(
                    record.benchmark, record.stage2_row,
                    record.stage2_row["output"],
                ))
                for record in records
            ),
        },
        "criteria": {
            "minimum_net_gain": criteria.minimum_net_gain,
            "max_mean_observations": criteria.max_mean_observations,
            "preferred_mean_observations": criteria.preferred_mean_observations,
            "all_cells_nonnegative": True,
            "both_backbones_strictly_positive": True,
            "corrections_exceed_corruptions": True,
        },
        "selection": selection_payload,
        "selection_metrics_kind": "four-fold source-group OOF",
        "selection_mean_observations": selection.metrics.observations / len(records),
        "deployed_refit_replay": {
            "metrics": refit_metrics.to_dict(),
            "failures": list(refit_failures),
            "mean_observations": refit_metrics.observations / len(records),
        },
        "policy_sha256": _canonical_sha256(policy_payload),
        "policy": policy_payload,
        "method_contract": {
            "inference_fields": [
                "source image", "question", "visible options",
                "Stage-3 observation trajectory", "candidate-free verifier answer",
            ],
            "forbidden_inference_fields": [
                "answer", "correctness", "dataset identity", "backbone identity",
                "category", "ground-truth geometry",
            ],
            "shared_rule": True,
            "exact_fallback": "P0",
        },
        "inputs": {
            "verifier_observations": observation_audits,
            "support_calibration": {
                "path": str(Path(calibration_path)),
                "sha256": _sha256(Path(calibration_path)),
            },
            "code": code_manifest,
        },
        "mme_realworld_lite_outcomes_read": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = freeze_policy()
    encoded = json.dumps(
        payload, indent=2, ensure_ascii=False, allow_nan=False,
    ) + "\n"
    if args.output is None:
        print(encoded, end="")
        return 0
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
