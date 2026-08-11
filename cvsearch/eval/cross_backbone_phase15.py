"""Replay the frozen Phase15 policy on a model-neutral combined pair."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cvsearch.eval.phase2_oracle import load_jsonl, load_launch_manifest
import cvsearch.eval.phase3_zoom_oracle as phase3
import cvsearch.eval.phase4_expand_oracle as phase4
import cvsearch.eval.phase6_combined_selection as phase6
from cvsearch.eval.phase6_combined_selection import ExtractedCombinedPair
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
_INPUT_FIELDS = ("input_image", "question", "options", "answer_type")


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


def validate_and_extract_cross_backbone_pairs(
    benchmark: str,
    disabled_rows: Sequence[Mapping[str, Any]],
    enabled_rows: Sequence[Mapping[str, Any]],
    disabled_manifest: Mapping[str, Any],
    enabled_manifest: Mapping[str, Any],
    *, expected_inference_revision: str,
) -> tuple[ExtractedCombinedPair, ...]:
    """Validate paired provenance and model-neutral action DTO boundaries."""
    provenance = phase6.validate_combined_launch_pair(
        disabled_rows, enabled_rows, disabled_manifest, enabled_manifest,
        expected_inference_revision=expected_inference_revision,
    )
    if provenance.benchmark != benchmark:
        raise ValueError("requested benchmark differs from paired launch")
    disabled_config, enabled_config = phase6._validate_config_pair(
        phase3._mapping(disabled_manifest["config"]["loaded"], "disabled config"),
        phase3._mapping(enabled_manifest["config"]["loaded"], "enabled config"),
    )
    result = []
    for disabled, enabled in zip(disabled_rows, enabled_rows):
        ordinal = disabled.get("_eg_ordinal")
        if any(disabled.get(field) != enabled.get(field) for field in _INPUT_FIELDS):
            raise ValueError("paired rows differ in exact input identity")
        expected_type = "logits_match" if benchmark == "vstar" else "option_list"
        output = disabled.get("output")
        if disabled.get("answer_type") != expected_type or enabled.get("output") != output:
            raise ValueError("paired rows differ in answer type or public P0")
        disabled_trace = phase3._mapping(disabled.get("method_trace"), "disabled trace")
        enabled_trace = phase3._mapping(enabled.get("method_trace"), "enabled trace")
        if any(
            disabled_trace.get(field) != enabled_trace.get(field)
            for field in phase4._SHARED_P0_TRACE_FIELDS
        ):
            raise ValueError("paired traces differ in shared P0 identity")
        options = phase3._list(enabled.get("options"), "paired options")
        _, _, disabled_answer, _ = phase4._validate_common_trace(
            disabled_trace, config=disabled_config, enabled=False,
            benchmark=benchmark, options=options, output=output,
            context=f"disabled row {ordinal}",
        )
        _, _, enabled_answer, steps = phase6._validate_enabled_trace(
            enabled_trace, config=enabled_config, benchmark=benchmark,
            options=options, output=output, context=f"enabled row {ordinal}",
        )
        if disabled_answer != enabled_answer:
            raise ValueError("paired traces differ in exact P0 answer")
        zoom = phase3._mapping(steps[0].get("zoom_audit"), "ZOOM audit")
        expand = phase3._mapping(steps[1].get("expand_audit"), "EXPAND audit")
        if set(zoom) != set(phase3._AUDIT_FIELDS) or set(expand) != set(phase4._AUDIT_FIELDS):
            raise ValueError("cross-backbone action audit schema is invalid")
        base_size = zoom.get("base_view_size")
        candidate_size = zoom.get("candidate_view_size")
        if (
            type(base_size) is not int or base_size <= 0
            or type(candidate_size) is not int
            or candidate_size != base_size // 3
        ):
            raise ValueError("ZOOM sizes are not derived from the native backbone")
        if (
            zoom.get("p0_anchor") != expand.get("p0_anchor")
            or zoom.get("p0_stability") != expand.get("p0_stability")
        ):
            raise ValueError("actions do not share one exact P0 audit")
        p0_stability = phase3._mapping(zoom.get("p0_stability"), "P0 stability")
        if p0_stability.get("output") != output:
            raise ValueError("P0 audit output differs from public P0")
        p0 = {
            "action": "P0", "output": json.loads(phase6._snapshot(output)),
            "p0_stability": {
                "confidence": phase6._confidence(p0_stability, "P0 confidence"),
            },
        }
        candidates = (
            phase6._candidate_dto("ZOOM", zoom),
            phase6._candidate_dto("EXPAND", expand),
        )
        input_identity = {
            "ordinal": ordinal,
            **{
                field: json.loads(phase6._snapshot(enabled.get(field)))
                for field in _INPUT_FIELDS
            },
        }
        result.append(ExtractedCombinedPair(
            ordinal=ordinal, provenance=provenance,
            _input_identity_json=phase6._snapshot(input_identity),
            _p0_json=phase6._snapshot(p0),
            _candidate_jsons=tuple(phase6._snapshot(item) for item in candidates),
        ))
    return tuple(result)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_rows = load_jsonl(args.base_jsonl)
    combined_rows = load_jsonl(args.combined_jsonl)
    base_manifest = load_launch_manifest(args.base_jsonl)
    combined_manifest = load_launch_manifest(args.combined_jsonl)
    revision = base_manifest["code"]["revision"]
    pairs = validate_and_extract_cross_backbone_pairs(
        args.benchmark, base_rows, combined_rows, base_manifest, combined_manifest,
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
