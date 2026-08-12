"""CPU-only replay of logged V* candidate score components."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cvsearch.evidence_gap.query_profile import adaptive_alpha, infer_query_profile

from .eval_vstar_ranking_recall import _load_jsonl, evaluate_rows, group_rank_events


def _weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _component(detail: Mapping[str, Any], name: str) -> float:
    score = detail.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("candidate detail score must be a mapping")
    value = score.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"candidate {name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"candidate {name} must be finite")
    return result


def fused_rank(
    detail: Mapping[str, Any], *, beta: float, alpha: float, visual_lambda: float,
) -> float:
    beta = _weight(beta, "beta")
    alpha = _weight(alpha, "alpha")
    visual_lambda = _weight(visual_lambda, "visual_lambda")
    relevance = beta * _component(detail, "main") + (1.0 - beta) * _component(
        detail, "augmented"
    )
    visual = visual_lambda * _component(detail, "complexity") + (
        1.0 - visual_lambda
    ) * _component(detail, "edge_density")
    return alpha * relevance + (1.0 - alpha) * visual


def replay_event(
    event: Sequence[Mapping[str, Any]], *, beta: float, alpha: float, visual_lambda: float,
) -> list[Mapping[str, Any]]:
    """Return a stable score-sorted copy of one logged rank event."""
    indexed = list(enumerate(event))
    indexed.sort(key=lambda item: (
        -fused_rank(
            item[1], beta=beta, alpha=alpha, visual_lambda=visual_lambda,
        ),
        item[0],
    ))
    return [detail for _, detail in indexed]


def _event_query_profile(
    row: Mapping[str, Any], event: Sequence[Mapping[str, Any]],
):
    """Mirror the query inputs constructed by CVSearch._make_rank_context."""
    if not event:
        raise ValueError("rank event must not be empty")
    question = row.get("question")
    trace = row.get("method_trace")
    plan = trace.get("query_plan") if isinstance(trace, Mapping) else None
    planned_main = plan.get("main_query") if isinstance(plan, Mapping) else None
    main_query = (
        planned_main
        if isinstance(planned_main, str) and planned_main.strip()
        else question
    )
    augmented = plan.get("augmented_queries") if isinstance(plan, Mapping) else None
    if not isinstance(augmented, (list, tuple)) or not augmented:
        augmented = [event[0].get("target")]
    return infer_query_profile(main_query, augmented)


def _config(value: Mapping[str, Any]) -> dict[str, Any]:
    base_keys = {"name", "beta", "alpha", "visual_lambda"}
    adaptive_keys = {
        "detail_alpha_discount", "context_alpha_gain",
        "context_visual_discount",
    }
    if not isinstance(value, Mapping) or not base_keys.issubset(value) or not set(value).issubset(
        base_keys | adaptive_keys
    ):
        raise ValueError("replay config has invalid keys")
    name = value["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("replay config name must be nonempty")
    result = {
        "name": name,
        "beta": _weight(value["beta"], "beta"),
        "alpha": _weight(value["alpha"], "alpha"),
        "visual_lambda": _weight(value["visual_lambda"], "visual_lambda"),
        "detail_alpha_discount": _weight(
            value.get("detail_alpha_discount", 0.0), "detail_alpha_discount"
        ),
        "context_alpha_gain": _weight(
            value.get("context_alpha_gain", 0.0), "context_alpha_gain"
        ),
    }
    if "context_visual_discount" in value:
        result["context_visual_discount"] = _weight(
            value["context_visual_discount"], "context_visual_discount"
        )
    return result


def replay_rows(
    rows: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    *,
    k: int = 3,
) -> dict[str, Any]:
    """Replay complete rows while preserving their candidate pools and source bytes."""
    if k != 3:
        raise ValueError("v1 replay keeps the declared K=3 fixed")
    normalized = [_config(config) for config in configs]
    if not normalized or len({config["name"] for config in normalized}) != len(normalized):
        raise ValueError("replay configs must have unique nonempty names")
    candidates = []
    for config in normalized:
        replayed = copy.deepcopy(list(rows))
        for row in replayed:
            trace = row.get("method_trace")
            if not isinstance(trace, dict):
                raise ValueError("row method_trace must be a dictionary")
            events = group_rank_events(trace.get("candidate_ranks", []))
            replayed_details = []
            for event in events:
                profile = _event_query_profile(row, event)
                alpha = adaptive_alpha(
                    config["alpha"],
                    profile,
                    config["detail_alpha_discount"],
                    config["context_alpha_gain"],
                )
                visual_lambda = min(1.0, max(
                    0.0,
                    config["visual_lambda"]
                    - config.get("context_visual_discount", 0.0)
                    * profile.context_demand,
                ))
                replayed_details.extend(replay_event(
                    event,
                    beta=config["beta"],
                    alpha=alpha,
                    visual_lambda=visual_lambda,
                ))
            trace["candidate_ranks"] = replayed_details
        candidates.append({"config": config, "metrics": evaluate_rows(replayed)})
    report = {"schema_version": 1, "k": k, "candidates": candidates}
    json.dumps(report, allow_nan=False)
    return report


def _grid() -> list[dict[str, Any]]:
    values = (0.0, 0.25, 0.5, 0.75, 1.0)
    adaptations = (0.0, 0.15, 0.3, 0.45)
    configs = [{
        "name": "current",
        "beta": 0.6,
        "alpha": 0.65,
        "visual_lambda": 0.5,
        "detail_alpha_discount": 0.0,
        "context_alpha_gain": 0.0,
    }]
    configs.extend({
        "name": (
            f"b{beta:.2f}-a{alpha:.2f}-v{visual_lambda:.2f}"
            f"-d{detail_discount:.2f}-c{context_gain:.2f}"
        ),
        "beta": beta,
        "alpha": alpha,
        "visual_lambda": visual_lambda,
        "detail_alpha_discount": detail_discount,
        "context_alpha_gain": context_gain,
    } for beta, alpha, visual_lambda, detail_discount, context_gain in itertools.product(
        values, values, values, adaptations, adaptations,
    ))
    return configs


def _metric(candidate: Mapping[str, Any], test_type: str | None, k: str) -> float:
    metrics = candidate["metrics"]
    if test_type is not None:
        metrics = metrics["by_test_type"][test_type]
    value = metrics["topic"]["recall_at"][k]
    return -1.0 if value is None else float(value)


def _select(report: dict[str, Any]) -> None:
    baseline = report["candidates"][0]
    test_types = tuple(baseline["metrics"]["by_test_type"])
    eligible = [candidate for candidate in report["candidates"] if all(
        _metric(candidate, test_type, "3") >= _metric(baseline, test_type, "3")
        for test_type in test_types
    )]
    best = max(eligible, key=lambda candidate: (
        _metric(candidate, None, "3"),
        _metric(candidate, None, "1"),
        candidate["metrics"]["topic"]["median_top3_concentration"] or -1.0,
        candidate["config"]["name"],
    ))
    report["baseline"] = baseline
    report["best_non_regressing"] = best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)
    report = replay_rows(_load_jsonl(args.answers_file), _grid(), k=args.k)
    _select(report)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
