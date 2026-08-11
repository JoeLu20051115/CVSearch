"""Replay the frozen Phase15 policy on a model-neutral combined pair."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cvsearch.eval.phase2_oracle import load_jsonl, load_launch_manifest
from cvsearch.eval.phase6_combined_selection import (
    validate_and_extract_combined_pairs,
)
from cvsearch.eval.phase7_confirmation_runner import _atomic_write, _sha256_file
from cvsearch.eval.phase9_dense_evidence_runner import _load_clip
from cvsearch.eval.phase12_generated_query_lazy_runner import (
    _produce_record as produce_lazy_record,
    is_eligible_lazy_pair,
)
from cvsearch.eval.phase5_unified_selector import select_unified_state
from cvsearch.eval.phase14_hr_semantic_normalization import (
    select_hr_semantic_normalization,
)
from cvsearch.eval.phase15_vstar_confirmed_backtrack import (
    B8_FROZEN_B5_RULE,
    select_vstar_confirmed_backtrack,
)
from cvsearch.evidence_gap.provenance import canonical_sha256, content_manifest
from cvsearch.models.modeling_dispatch import load_search_model


_HR_BENCHMARKS = frozenset({"hr-bench_4k", "hr-bench_8k"})


@dataclass(frozen=True)
class CrossBackbonePhase15Decision:
    action: str
    status: str
    output: Any
    phase6_action: str


def select_phase15_output(
    benchmark: str,
    p0: dict[str, Any],
    candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    options: Sequence[str],
    lazy_record: Mapping[str, Any] | None = None,
) -> CrossBackbonePhase15Decision:
    """Apply Phase6 first, then only the benchmark-specific Phase15 fallback."""
    phase6 = select_unified_state(p0, candidates)
    if phase6.action != "P0":
        return CrossBackbonePhase15Decision(
            phase6.action, phase6.status, phase6.output, phase6.action,
        )
    if benchmark in _HR_BENCHMARKS:
        selected = select_hr_semantic_normalization(p0, options)
    elif benchmark == "vstar" and lazy_record is not None:
        try:
            b5_decision = lazy_record["decisions"][B8_FROZEN_B5_RULE]
            observations = lazy_record["observations"]
        except (KeyError, TypeError):
            b5_decision = {}
            observations = None
        selected = select_vstar_confirmed_backtrack(
            p0, options, observations, b5_decision,
        )
    elif benchmark == "vstar":
        selected = phase6
    else:
        raise ValueError(f"unsupported benchmark: {benchmark!r}")
    return CrossBackbonePhase15Decision(
        selected.action, selected.status, selected.output, phase6.action,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True,
        choices=("vstar", "hr-bench_4k", "hr-bench_8k"),
    )
    parser.add_argument("--base-jsonl", type=Path, required=True)
    parser.add_argument("--combined-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _artifact_manifest(path: Path) -> dict[str, Any]:
    allowed_root = path.parents[1] if path.parent.name == "snapshots" else None
    return content_manifest(path, allowed_symlink_root=allowed_root)


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
    if len(rows) != len(base_rows) or set(rows) != {pair.ordinal for pair in pairs}:
        raise ValueError("Phase15 source rows do not exactly cover validated pairs")

    model_artifact = _artifact_manifest(args.model_path)
    expected_model = combined_manifest["artifacts"]["qwen"]["sha256"]
    if (
        model_artifact["sha256"] != expected_model
        or base_manifest["artifacts"]["qwen"]["sha256"] != expected_model
    ):
        raise ValueError("Phase15 model artifact differs from paired source runs")

    eligible = [
        pair for pair in pairs
        if args.benchmark == "vstar"
        and is_eligible_lazy_pair(pair, rows[pair.ordinal])
    ]
    clip_artifact = _artifact_manifest(args.clip_model_path)
    model = load_search_model(args.model_path, device="cuda:0") if eligible else None
    clip_model, clip_processor = (
        _load_clip(args.clip_model_path) if eligible else (None, None)
    )
    lazy_records = [
        produce_lazy_record(
            pair, rows[pair.ordinal], base_manifest, model,
            clip_model, clip_processor,
        )
        for pair in eligible
    ]
    lazy_by_ordinal = {record["source_ordinal"]: record for record in lazy_records}
    if len(lazy_by_ordinal) != len(lazy_records):
        raise ValueError("Phase15 lazy records contain duplicate ordinals")

    decisions = [
        select_phase15_output(
            args.benchmark, pair.p0, pair.candidates,
            rows[pair.ordinal]["options"], lazy_by_ordinal.get(pair.ordinal),
        )
        for pair in pairs
    ]
    output_rows = []
    for pair, decision in zip(pairs, decisions):
        row = dict(rows[pair.ordinal])
        row["output"] = decision.output
        row["_cross_backbone_phase15"] = {
            **asdict(decision),
            "extracted_pair_sha256": pair.extracted_digest,
            "lazy_record_sha256": (
                canonical_sha256(lazy_by_ordinal[pair.ordinal])
                if pair.ordinal in lazy_by_ordinal else None
            ),
        }
        output_rows.append(row)

    lazy_path = Path(f"{args.output}.lazy.jsonl")
    lazy_sha256 = _atomic_write(lazy_path, lazy_records)
    output_sha256 = _atomic_write(args.output, output_rows)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "cross-backbone-phase15",
        "benchmark": args.benchmark,
        "inference_revision": revision,
        "base_jsonl": {
            "path": str(args.base_jsonl), "sha256": _sha256_file(args.base_jsonl),
            "manifest_sha256": canonical_sha256(base_manifest),
        },
        "combined_jsonl": {
            "path": str(args.combined_jsonl),
            "sha256": _sha256_file(args.combined_jsonl),
            "manifest_sha256": canonical_sha256(combined_manifest),
        },
        "model_artifact": model_artifact,
        "clip_artifact": clip_artifact,
        "selectors": {
            "phase6": _sha256_file(Path(__file__).with_name("phase5_unified_selector.py")),
            "hr_b7": _sha256_file(
                Path(__file__).with_name("phase14_hr_semantic_normalization.py")
            ),
            "vstar_b8": _sha256_file(
                Path(__file__).with_name("phase15_vstar_confirmed_backtrack.py")
            ),
            "vstar_frozen_b5_rule": B8_FROZEN_B5_RULE,
        },
        "records": len(output_rows),
        "ordinals": [pair.ordinal for pair in pairs],
        "eligible_lazy_records": len(eligible),
        "lazy_failures": sum(record["failure"] is not None for record in lazy_records),
        "action_counts": dict(sorted(Counter(
            decision.action for decision in decisions
        ).items())),
        "selected_output_sha256": canonical_sha256([
            decision.output for decision in decisions
        ]),
        "lazy_jsonl": {"path": str(lazy_path), "sha256": lazy_sha256},
        "output_sha256": output_sha256,
        "runner_source_sha256": _sha256_file(Path(__file__)),
    }
    manifest_path = Path(f"{args.output}.phase15-manifest.json")
    _atomic_write(manifest_path, [manifest])
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
