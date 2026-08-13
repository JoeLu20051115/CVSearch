#!/usr/bin/env python3
"""Append Stage 3 split observations to frozen Stage 2 JSONL rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from cvsearch.evidence_gap.clip_scorer import ClipScorer
from cvsearch.evidence_gap.method import _BudgetedZoomModel, _observe_split_search
from cvsearch.evidence_gap.types import (
    AnswerRecord,
    BudgetLedger,
    P0Anchor,
    QueryPlan,
)


_SPLIT_POLICY_BUDGETS = {
    "native_2x2_overlap_support_screen_two_scale_depth2_v2": 4,
    "native_2x2_overlap_support_screen_three_scale_all_roots_depth2_v3": 6,
}


def split_policy_budget(render_policy: str) -> tuple[str, int]:
    if render_policy not in _SPLIT_POLICY_BUDGETS:
        raise ValueError("split runner render policy is not frozen")
    return render_policy, _SPLIT_POLICY_BUDGETS[render_policy]


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL row {line_number}") from error
        if type(row) is not dict:
            raise ValueError("split input rows must be exact JSON objects")
        rows.append(row)
    if not rows:
        raise ValueError("split input JSONL must be nonempty")
    return rows


def _query_plan(row: Mapping[str, Any]) -> QueryPlan:
    trace = row.get("method_trace")
    value = trace.get("query_plan") if isinstance(trace, Mapping) else None
    if type(value) is not dict or set(value) != {
        "main_query", "targets", "augmented_queries", "evidence_items",
        "global_scope_required", "fallback_used",
    }:
        raise ValueError("frozen row has no exact query plan")
    return QueryPlan(
        main_query=value["main_query"],
        targets=tuple(value["targets"]),
        augmented_queries=tuple(value["augmented_queries"]),
        evidence_items=tuple(value["evidence_items"]),
        global_scope_required=value["global_scope_required"],
        fallback_used=value["fallback_used"],
    )


def _p0_stability(row: Mapping[str, Any]) -> AnswerRecord:
    trace = row.get("method_trace")
    value = trace.get("final_answer") if isinstance(trace, Mapping) else None
    if type(value) is not dict:
        raise ValueError("frozen row has no final answer record")
    return AnswerRecord(
        output=copy.deepcopy(row.get("output")),
        canonical_answer=copy.deepcopy(value.get("canonical_answer")),
        raw_outputs=tuple(copy.deepcopy(value.get("raw_outputs", ()))),
        groups=copy.deepcopy(value.get("groups", {})),
        frequency=value.get("frequency", 0.0),
        margin=value.get("margin", 0.0),
        confidence=value.get("confidence", 0.0),
        uncertainty=value.get("uncertainty", 1.0),
        losses=tuple(value.get("losses", ())),
        selected_from=value.get("selected_from", "stage2_frozen"),
        aggregation_available=value.get("aggregation_available"),
        aggregation_reason=value.get("aggregation_reason"),
    )


def _model_family(model_path: Path) -> str:
    value = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    model_type = value.get("model_type")
    if model_type in {"qwen2_5_vl", "qwen3_vl"}:
        return "qwen"
    if model_type == "internvl_chat":
        return "internvl"
    if model_type == "llava":
        return "llava_verifier"
    raise ValueError(f"unsupported Stage-3 model type: {model_type!r}")


def _load_model(model_path: Path):
    family = _model_family(model_path)
    if family == "qwen":
        from cvsearch.models.modeling_qwenvl import ModelQwenVL
        return ModelQwenVL(
            model_path=str(model_path), device="cuda:0",
            torch_dtype=torch.bfloat16, patch_scale=1.2,
        )
    if family == "llava_verifier":
        from cvsearch.models.modeling_llava import ModelGlobalLocal
        return ModelGlobalLocal(
            model_path=str(model_path), conv_type="qwen_1_5",
            device="cuda:0", torch_dtype=torch.bfloat16, patch_scale=1.2,
        )
    from cvsearch.models.modeling_internvl import ModelInternvl
    return ModelInternvl(
        model_path=str(model_path), device="cuda:0",
        torch_dtype=torch.bfloat16, patch_scale=1.2,
    )


def _source_image(row: Mapping[str, Any], image_root: Path) -> Image.Image:
    value = row.get("input_image")
    if not isinstance(value, str) or not value:
        raise ValueError("split row input_image must be nonempty")
    path = image_root / value
    with Image.open(path) as image:
        return image.convert("RGB")


def _observe_row(
    row: dict[str, Any], *, model: Any, scorer: Any, image_root: Path,
    render_policy: str = (
        "native_2x2_overlap_support_screen_two_scale_depth2_v2"
    ),
) -> dict[str, Any]:
    trace = row.get("method_trace")
    if type(trace) is not dict or type(trace.get("candidate_ranks")) is not list:
        raise ValueError("split row lacks frozen Stage-1 ranks")
    if any(
        isinstance(step, Mapping) and step.get("action") == "SPLIT"
        for step in trace.get("steps", [])
    ):
        raise ValueError("split row already contains a SPLIT observation")
    source = _source_image(row, image_root)
    ledger = BudgetLedger(512, 10_000_000_000)
    wrapped = _BudgetedZoomModel(model, ledger)
    p0 = _p0_stability(row)
    anchor = P0Anchor(
        emitted_answer=copy.deepcopy(row.get("output")),
        cvsearch_raw=copy.deepcopy(row.get("output")),
        producing_phase="response",
        node_keys=(),
        support_view=None,
    )
    started = time.perf_counter()
    render_policy, branch_budget = split_policy_budget(render_policy)
    audit = _observe_split_search(
        budgeted_model=wrapped,
        source_image=source,
        scorer=scorer,
        query_plan=_query_plan(row),
        answer_type=row.get("answer_type"),
        options=row.get("options"),
        p0_anchor=anchor,
        p0_stability=p0,
        stage1_rank_sha256=_sha256(trace["candidate_ranks"]),
        render_policy=render_policy,
        max_observed_branches=branch_budget,
    )
    result = copy.deepcopy(row)
    result["method_trace"]["steps"].append({
        "step": len(result["method_trace"]["steps"]),
        "action": "SPLIT",
        "split_search_audit": audit.to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
    })
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    payload = "".join(_canonical(row) + "\n" for row in rows)
    partial.write_text(payload, encoding="utf-8")
    partial.replace(path)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--render-policy", choices=tuple(_SPLIT_POLICY_BUDGETS),
        default="native_2x2_overlap_support_screen_two_scale_depth2_v2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _load_jsonl(args.input_jsonl)
    model = _load_model(args.model_path)
    scorer = ClipScorer(device="cuda:0", model_path=str(args.clip_model_path))
    output = [
        _observe_row(
            row, model=model, scorer=scorer, image_root=args.image_root,
            render_policy=args.render_policy,
        )
        for row in rows
    ]
    output_sha256 = _write_jsonl(args.output, output)
    manifest = {
        "schema_version": 1,
        "artifact_kind": "stage3-split-candidate-observations",
        "model_family": _model_family(args.model_path),
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": hashlib.sha256(args.input_jsonl.read_bytes()).hexdigest(),
        "model_path": str(args.model_path),
        "clip_model_path": str(args.clip_model_path),
        "image_root": str(args.image_root),
        "render_policy": args.render_policy,
        "max_observed_branches": split_policy_budget(args.render_policy)[1],
        "ordinals": [row["_eg_ordinal"] for row in rows],
        "records": len(output),
        "output_sha256": output_sha256,
    }
    manifest_path = Path(str(args.output) + ".split-manifest.json")
    _write_jsonl(manifest_path, [manifest])
    print(_canonical(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
