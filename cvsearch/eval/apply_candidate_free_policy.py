#!/usr/bin/env python3
"""Apply one frozen robust-transfer policy without reading evaluator labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cvsearch.eval.analyze_split_search import load_selected_calibrations
from cvsearch.eval.candidate_free_verifier_selector import (
    CandidateFreeRuntimeEvidence,
    candidate_free_runtime_outcome,
)
from cvsearch.eval.freeze_candidate_free_policy import (
    _canonical_sha256,
    _sha256,
    verified_pairwise_manifest,
)
from cvsearch.eval.replay_adaptive_search import replay_adaptive_search
from cvsearch.eval.replay_uncertainty_support import (
    RiskCalibrator,
    UnifiedPolicy,
    UtilityIsotonicCalibrator,
    _prepare_replay,
    candidate_snapshots,
    sanitize_replay_row,
)


def runtime_evidence_from_record(
    record: Mapping[str, Any],
) -> CandidateFreeRuntimeEvidence:
    """Project a verifier record while ignoring every evaluator-only field."""
    if not isinstance(record, Mapping):
        raise TypeError("runtime verifier record must be a mapping")
    proposal = record.get("proposal")
    projection = record.get("projection")
    cost = record.get("cost")
    if type(proposal) is not dict or type(cost) is not dict:
        raise ValueError("runtime verifier proposal or cost is missing")
    proposal_feasible = proposal.get("feasible")
    if type(proposal_feasible) is not bool:
        raise ValueError("runtime proposal feasibility is invalid")
    if projection is not None and type(projection) is not dict:
        raise ValueError("runtime verifier projection is invalid")
    verifier_feasible = bool(projection and projection.get("feasible") is True)
    if proposal_feasible and (
        proposal.get("candidate_output") is None
        or proposal.get("candidate_canonical") is None
    ):
        raise ValueError("feasible runtime proposal is incomplete")
    if verifier_feasible and projection.get("canonical_answer") is None:
        raise ValueError("feasible runtime verifier projection is incomplete")
    return CandidateFreeRuntimeEvidence(
        verifier_feasible=verifier_feasible,
        verifier_canonical=(
            projection.get("canonical_answer") if projection else None
        ),
        verifier_confidence=(
            projection.get("confidence", 0.0) if projection else 0.0
        ),
        proposal_feasible=proposal_feasible,
        proposal_output=copy.deepcopy(proposal.get("candidate_output")),
        proposal_canonical=copy.deepcopy(proposal.get("candidate_canonical")),
        proposal_agreement=proposal.get("agreement", 0.0),
        observations=cost.get("observations"),
    )


def _load_jsonl(path: Path, ordinal_field: str) -> dict[int, dict[str, Any]]:
    result = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            ordinal = row.get(ordinal_field) if type(row) is dict else None
            if type(ordinal) is not int or ordinal < 0 or ordinal in result:
                raise ValueError(f"invalid or duplicate ordinal at {path}:{line_number}")
            result[ordinal] = row
    if not result:
        raise ValueError(f"runtime JSONL is empty: {path}")
    return result


def _load_policy(path: Path) -> tuple[dict[str, Any], RiskCalibrator]:
    with Path(path).open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    if (
        type(artifact) is not dict
        or artifact.get("artifact_kind")
        != "robust-transfer-candidate-free-policy"
        or artifact.get("status") != "frozen_before_mme_outcomes"
    ):
        raise ValueError("runtime policy artifact is not frozen")
    policy = artifact.get("policy")
    if type(policy) is not dict or artifact.get("policy_sha256") != _canonical_sha256(policy):
        raise ValueError("runtime policy hash is invalid")
    calibrators = policy.get("refit_calibrators")
    if type(calibrators) is not dict or set(calibrators) != {"*/*"}:
        raise ValueError("runtime policy must contain one global refit calibrator")
    return artifact, RiskCalibrator.from_dict(calibrators["*/*"])


def _p0_output(stage2: Mapping[str, Any], calibration: Any) -> Any:
    fallback = replay_adaptive_search(
        sanitize_replay_row(stage2), sanitize_replay_row(stage2), calibration,
    )
    return copy.deepcopy(fallback["selected_output"])


def apply_runtime_record(
    stage2_row: Mapping[str, Any],
    split_row: Mapping[str, Any],
    verifier_row: Mapping[str, Any],
    calibration: Any,
    artifact: Mapping[str, Any],
    calibrator: RiskCalibrator,
) -> dict[str, Any]:
    """Return one label-blind decision bound to the frozen policy hash."""
    evidence = runtime_evidence_from_record(verifier_row)
    p0 = _p0_output(stage2_row, calibration)
    failure = None
    aggregate_unavailable = None
    try:
        prepared = _prepare_replay(stage2_row, split_row, calibration)
        p0 = copy.deepcopy(prepared.stage2["selected_output"])
        dummy = UnifiedPolicy(
            profile="balanced",
            threshold=0.0,
            raw_support_floor=0.0,
            utility_calibrator=UtilityIsotonicCalibrator((1.0,), (0.5,)),
        )
        snapshots = candidate_snapshots(
            stage2_row, split_row, calibration, dummy,
        )
    except (KeyError, TypeError, ValueError) as error:
        snapshots = ()
        aggregate_unavailable = {
            "type": type(error).__name__,
            "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        }
    try:
        policy = artifact["policy"]
        outcome = candidate_free_runtime_outcome(
            snapshots, p0, calibrator, evidence,
            veto_confidence=policy["aggregate_veto_confidence"],
            proposal_confidence=policy["proposal_verifier_confidence"],
            proposal_agreement=policy["proposal_agreement"],
        )
    except (KeyError, TypeError, ValueError) as error:
        failure = {
            "type": type(error).__name__,
            "message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
        }
        outcome = None
    if outcome is None:
        selected_output = p0
        selected_source = "P0"
        selected_canonical = None
        observations = evidence.observations
        vetoed = False
    else:
        selected_output = outcome.selected_output
        selected_source = outcome.selected_source
        selected_canonical = outcome.selected_canonical
        observations = outcome.observations
        vetoed = outcome.vetoed
    result = {
        "source_ordinal": stage2_row.get("_eg_ordinal"),
        "input_image": stage2_row.get("input_image"),
        "p0_output": p0,
        "selected_output": selected_output,
        "selected_canonical": selected_canonical,
        "selected_source": selected_source,
        "observations": observations,
        "vetoed": vetoed,
        "policy_sha256": artifact["policy_sha256"],
        "aggregate_unavailable": aggregate_unavailable,
        "failure": failure,
    }
    json.dumps(result, allow_nan=False)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact, calibrator = _load_policy(args.policy)
    calibrations = load_selected_calibrations(args.support_calibration)
    calibration = calibrations[args.backbone]
    stage2 = _load_jsonl(args.stage2, "_eg_ordinal")
    split = _load_jsonl(args.split, "_eg_ordinal")
    verifier = _load_jsonl(args.verifier, "source_ordinal")
    verified_pairwise_manifest(args.verifier)
    if stage2.keys() != split.keys() or stage2.keys() != verifier.keys():
        raise ValueError("runtime Stage-2/SPLIT/verifier ordinals differ")
    for ordinal in stage2:
        if (
            verifier[ordinal].get("backbone") != args.backbone
            or verifier[ordinal].get("benchmark") != "mme-realworld-lite"
            or stage2[ordinal].get("input_image") != split[ordinal].get("input_image")
        ):
            raise ValueError(f"runtime identity mismatch at ordinal {ordinal}")
    output = args.output.resolve()
    manifest_path = Path(f"{output}.runtime-manifest.json")
    if output.exists() or output.is_symlink() or manifest_path.exists():
        raise FileExistsError(output)
    rows = [
        apply_runtime_record(
            stage2[ordinal], split[ordinal], verifier[ordinal], calibration,
            artifact, calibrator,
        )
        for ordinal in sorted(stage2)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{output}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    payload = "".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        for row in rows
    )
    partial.write_text(payload, encoding="utf-8")
    os.replace(partial, output)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "robust-transfer-external-decisions",
        "data_scope": "mme-realworld-lite-label-blind",
        "backbone": args.backbone,
        "policy_path": str(args.policy.resolve()),
        "policy_file_sha256": _sha256(args.policy),
        "policy_sha256": artifact["policy_sha256"],
        "stage2_sha256": _sha256(args.stage2),
        "split_sha256": _sha256(args.split),
        "verifier_sha256": _sha256(args.verifier),
        "support_calibration_sha256": _sha256(args.support_calibration),
        "records": len(rows),
        "selections": sum(row["selected_source"] != "P0" for row in rows),
        "failures": sum(row["failure"] is not None for row in rows),
        "observations": sum(row["observations"] for row in rows),
        "output_sha256": _sha256(output),
    }
    manifest_path.write_text(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--support-calibration", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--backbone", choices=("qwen", "internvl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    manifest = run(build_parser().parse_args(argv))
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
