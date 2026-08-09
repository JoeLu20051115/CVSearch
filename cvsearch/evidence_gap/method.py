"""Leak-free composition for the first executable evidence-gap runtime."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .answers import aggregate_hr_answers, aggregate_vstar_losses
from .input import POLICY_FIELDS
from .policy import select_root_or_search
from .ranking import QueryAwareNodeRanker
from .types import (
    FORCED_RETURN,
    AnswerRecord,
    BudgetExceeded,
    BudgetLedger,
    HistoryRecord,
    MethodTrace,
    QueryPlan,
    StepTrace,
)


MINIMAL_V1: dict[str, Any] = {
    "config_id": "minimal_v1",
    "mode": "rerank_only",
    "rerank_enabled": True,
    "beta": 0.6,
    "alpha": 0.65,
    "visual_lambda": 0.5,
    "quick_gate": 0.8,
    "root_fallback_tolerance": 0.05,
    "enable_zoom": False,
    "enable_split": False,
    "enable_expand": False,
    "enable_certified_stop": False,
    "max_mllm_calls": 256,
    "max_processed_pixels": 10_000_000_000,
    "pixel_accounting": "source_image_area_per_logical_forward_approximation",
}

_GLOBAL_SCOPE = re.compile(
    r"\b(how many|number of|all|every|none|no|without|absent|only|unique|each)\b",
    re.IGNORECASE,
)
_RELATION = re.compile(
    r"\b(beside|between|behind|in front of|left of|right of|near|next to|above|below)\b",
    re.IGNORECASE,
)


def _strict_json(value: Any, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON-safe") from error


def _outputs_agree(left: Any, right: Any) -> bool:
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("evaluator output must be strict JSON-safe") from error


def _policy_copy(annotation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(annotation, Mapping):
        raise TypeError("policy_annotation must be a mapping")
    if set(annotation) != set(POLICY_FIELDS):
        raise ValueError(f"policy_annotation must contain exactly {list(POLICY_FIELDS)}")
    result = copy.deepcopy({key: annotation[key] for key in POLICY_FIELDS})
    if not isinstance(result["question"], str) or not result["question"].strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(result["answer_type"], str) or not result["answer_type"].strip():
        raise ValueError("answer_type must be a nonempty string")
    if not isinstance(result["input_image"], str) or not result["input_image"].strip():
        raise ValueError("input_image must be a nonempty string")
    _strict_json(result, "policy_annotation")
    return result


def build_query_plan(policy_annotation: Mapping[str, Any], targets: Sequence[str]) -> QueryPlan:
    """Build the deterministic, answer-free planner used by ``minimal_v1``."""
    if not isinstance(policy_annotation, Mapping):
        raise TypeError("policy_annotation must be a mapping")
    if set(policy_annotation) != set(POLICY_FIELDS):
        raise ValueError(f"policy_annotation must contain exactly {list(POLICY_FIELDS)}")
    question = policy_annotation["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence of strings")
    normalized_targets: list[str] = []
    seen: set[str] = set()
    for value in targets:
        if not isinstance(value, str):
            raise TypeError("targets must contain only strings")
        target = " ".join(value.split())
        if not target:
            continue
        key = target.casefold()
        if key not in seen:
            normalized_targets.append(target)
            seen.add(key)

    augmented = tuple(f"locate and inspect {target}" for target in normalized_targets)
    evidence_items: list[dict[str, Any]] = [
        {"kind": "target_detail", "target": target, "requirements": ["presence", "visual_detail"]}
        for target in normalized_targets
    ]
    if len(normalized_targets) > 1 or _RELATION.search(question):
        evidence_items.append({"kind": "relation_context", "targets": list(normalized_targets)})
    global_scope_required = bool(_GLOBAL_SCOPE.search(question))
    if global_scope_required:
        evidence_items.append({"kind": "coverage", "requirement": "global_scope"})
    if not evidence_items:
        evidence_items.append({"kind": "question_evidence", "requirement": "visual_detail"})

    plan = QueryPlan(
        main_query=question,
        targets=tuple(normalized_targets),
        augmented_queries=augmented,
        evidence_items=tuple(evidence_items),
        global_scope_required=global_scope_required,
        fallback_used=True,
    )
    _strict_json(plan.to_dict(), "query plan")
    return plan


def _finite_weight(config: dict[str, Any], name: str) -> None:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    config[name] = float(value)


def _nonnegative_integer(config: dict[str, Any], name: str) -> None:
    value = config[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _runtime_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _capture_runtime_diagnostics(trace: MethodTrace, runtime_annotation: Mapping[str, Any]) -> None:
    search_mode = runtime_annotation.get("search_mode")
    if search_mode is not None:
        if isinstance(search_mode, bool) or not isinstance(search_mode, int) or search_mode not in {0, 1, 2, 3}:
            raise ValueError("CVSearch search_mode must be one of 0, 1, 2, 3")
        trace.cvsearch_search_mode = search_mode
    root_confidence = runtime_annotation.get("root_ans_conf")
    if root_confidence is not None:
        trace.root_ans_conf = _runtime_number(root_confidence, "root_ans_conf")
    for key in ("num_pop", "num_zoom_in", "num_zoom_out"):
        value = runtime_annotation.get(key, [])
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"CVSearch {key} must be a sequence")
        snapshot = copy.deepcopy(list(value))
        _strict_json(snapshot, key)
        setattr(trace, key, snapshot)


def load_method_config(config: str | os.PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """Load a strict minimal-v1 config; deferred controller switches stay off."""
    if isinstance(config, Mapping):
        supplied = copy.deepcopy(dict(config))
    elif isinstance(config, (str, os.PathLike)):
        value = os.fspath(config)
        if value == "minimal_v1":
            supplied = {}
        else:
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8") as handle:
                supplied = json.load(handle)
            if not isinstance(supplied, dict):
                raise ValueError("config JSON must contain an object")
    else:
        raise TypeError("config must be a mapping, minimal_v1, or JSON path")
    unknown = set(supplied) - set(MINIMAL_V1)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    result = copy.deepcopy(MINIMAL_V1)
    result.update(supplied)
    if not isinstance(result["config_id"], str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", result["config_id"]
    ) is None:
        raise ValueError("config_id must be a nonempty version-safe identifier")
    if result["mode"] not in {"rerank_only", "root_search_fallback"}:
        raise ValueError("mode must be rerank_only or root_search_fallback")
    for name in (
        "rerank_enabled", "enable_zoom", "enable_split", "enable_expand", "enable_certified_stop"
    ):
        if not isinstance(result[name], bool):
            raise TypeError(f"{name} must be boolean")
    if any(result[name] for name in ("enable_zoom", "enable_split", "enable_expand", "enable_certified_stop")):
        raise ValueError("minimal_v1 cannot enable deferred actions or certification")
    for name in ("beta", "alpha", "visual_lambda", "quick_gate", "root_fallback_tolerance"):
        _finite_weight(result, name)
    for name in ("max_mllm_calls", "max_processed_pixels"):
        _nonnegative_integer(result, name)
    if result["pixel_accounting"] != "source_image_area_per_logical_forward_approximation":
        raise ValueError("pixel_accounting is fixed for minimal_v1")
    _strict_json(result, "method config")
    return result


class _BudgetedZoomModel:
    """Pre-charge public Qwen calls using source-image pixels as an approximation."""

    def __init__(self, model: Any, ledger: BudgetLedger, *, answer_reserve_calls: int = 0,
                 answer_image_loader: Any = None, answer_type: str | None = None):
        self._model = model
        self._ledger = ledger
        self._answer_reserve_calls = answer_reserve_calls
        self._answer_reserve_pixels: int | None = None if answer_reserve_calls else 0
        self._answer_image_loader = answer_image_loader
        self._answer_type = answer_type
        self._answer_started = False
        self._answer_reserve_invalid = False
        self._free_form_remaining = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    @staticmethod
    def _pixels(image: Any) -> int:
        if not isinstance(image, Image.Image):
            raise TypeError("visual model calls require a PIL image")
        return int(image.width * image.height)

    def _consume_actual(self, calls: int, pixels: int) -> None:
        """Atomically validate both dimensions before mutating the real ledger."""
        if self._ledger.mllm_calls + calls > self._ledger.max_mllm_calls:
            raise BudgetExceeded("mllm_calls budget exhausted before model execution")
        if self._ledger.processed_pixels + pixels > self._ledger.max_processed_pixels:
            raise BudgetExceeded("processed_pixels budget exhausted before model execution")
        self._ledger.consume("mllm_calls", calls)
        self._ledger.consume("processed_pixels", pixels)

    def _ensure_answer_reserve(self) -> None:
        if not self._answer_reserve_calls or self._answer_reserve_pixels is not None:
            return
        if self._ledger.mllm_calls + self._answer_reserve_calls > self._ledger.max_mllm_calls:
            self._answer_reserve_invalid = True
            raise BudgetExceeded("complete answer does not fit the mllm_calls budget")
        if self._answer_image_loader is None:
            raise AssertionError("answer reserve requires a source-image loader")
        source_image = self._answer_image_loader()
        self._answer_reserve_pixels = self._answer_reserve_calls * self._pixels(source_image)
        if self._ledger.processed_pixels + self._answer_reserve_pixels > self._ledger.max_processed_pixels:
            self._answer_reserve_invalid = True
            raise BudgetExceeded("complete answer does not fit the processed_pixels budget")

    @property
    def answer_reserve_available(self) -> bool:
        return (
            bool(self._answer_reserve_calls)
            and not self._answer_started
            and not self._answer_reserve_invalid
        )

    def _charge_search(self, calls: int, image: Image.Image | None = None) -> None:
        pixels = 0 if image is None else calls * self._pixels(image)
        self._ensure_answer_reserve()
        reserved_calls = self._answer_reserve_calls if not self._answer_started else 0
        reserved_pixels = int(self._answer_reserve_pixels or 0) if not self._answer_started else 0
        if self._ledger.mllm_calls + calls + reserved_calls > self._ledger.max_mllm_calls:
            raise BudgetExceeded("mllm_calls search admission would consume the final-answer reserve")
        if self._ledger.processed_pixels + pixels + reserved_pixels > self._ledger.max_processed_pixels:
            raise BudgetExceeded("processed_pixels search admission would consume the final-answer reserve")
        self._consume_actual(calls, pixels)

    def _charge_answer(self, calls: int, image: Image.Image) -> None:
        pixels = calls * self._pixels(image)
        if not self._answer_reserve_calls:
            self._consume_actual(calls, pixels)
            return
        self._ensure_answer_reserve()
        if self._answer_started:
            raise BudgetExceeded("the reserved final-answer operation was already consumed")
        if calls != self._answer_reserve_calls or pixels != self._answer_reserve_pixels:
            raise BudgetExceeded("final-answer operation does not match its reserved capacity")
        self._consume_actual(calls, pixels)
        self._answer_started = True

    def generate_visual_cues_using_ic(self, ic_examples: Any, question: str) -> Any:
        self._charge_search(1)
        return self._model.generate_visual_cues_using_ic(ic_examples, question)

    def get_confidence_value(self, nodes: Any, image_pil: Image.Image, *args: Any, **kwargs: Any) -> Any:
        self._charge_search(1, image_pil)
        return self._model.get_confidence_value(nodes, image_pil, *args, **kwargs)

    def free_form_using_nodes(self, image_pil: Image.Image, question: str, searched_nodes: Any, *args: Any,
                              **kwargs: Any) -> Any:
        if self._answer_reserve_calls:
            if self._answer_type != "option_list" or self._answer_reserve_calls != 4:
                raise AssertionError("reserved free-form calls require the four-block HR terminal answer")
            if not self._answer_started:
                self._charge_answer(4, image_pil)
                self._free_form_remaining = 4
            if self._free_form_remaining <= 0:
                raise BudgetExceeded("the reserved HR terminal batch was already consumed")
            result = self._model.free_form_using_nodes(
                image_pil, question, searched_nodes, *args, **kwargs
            )
            self._free_form_remaining -= 1
            return result
        self._charge_search(1, image_pil)
        return self._model.free_form_using_nodes(image_pil, question, searched_nodes, *args, **kwargs)

    def free_form_batch(self, image_pil: Image.Image,
                        requests: Sequence[tuple[str, Any]]) -> list[Any]:
        batch = list(requests)
        if not batch:
            raise ValueError("free-form batch must be nonempty")
        if self._answer_reserve_calls:
            self._charge_answer(len(batch), image_pil)
        else:
            self._charge_search(len(batch), image_pil)
        return [
            self._model.free_form_using_nodes(image_pil, question, searched_nodes)
            for question, searched_nodes in batch
        ]

    @staticmethod
    def _choice_calls(options: Any) -> int:
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or not options:
            raise ValueError("options must be a nonempty sequence")
        return 1 + len(options)

    def multiple_choices_inference(self, image_pil: Image.Image, question: str, options: Any,
                                   searched_nodes: Any = None) -> Any:
        self._charge_answer(self._choice_calls(options), image_pil)
        return self._model.multiple_choices_inference(image_pil, question, options, searched_nodes)

    def multiple_choices_with_losses(self, image_pil: Image.Image, question: str, options: Any,
                                     searched_nodes: Any = None) -> Any:
        self._charge_answer(self._choice_calls(options), image_pil)
        return self._model.multiple_choices_with_losses(image_pil, question, options, searched_nodes)


def _image_for(policy: Mapping[str, Any], image_folder: str | None) -> Image.Image:
    path = policy["input_image"]
    if image_folder is not None:
        path = os.path.join(image_folder, path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _as_answer_record(policy: Mapping[str, Any], raw: Any) -> AnswerRecord:
    answer_type = policy["answer_type"]
    if answer_type == "option_list":
        if not isinstance(policy["options"], list) or not isinstance(raw, list):
            raise ValueError("HR option_list requires option-block and output lists")
        return aggregate_hr_answers(policy["options"], raw)
    return AnswerRecord(output=copy.deepcopy(raw), canonical_answer=copy.deepcopy(raw))


def _root_answer(policy: Mapping[str, Any], model: _BudgetedZoomModel,
                 image_folder: str | None) -> AnswerRecord:
    image = _image_for(policy, image_folder)
    if policy["answer_type"] == "logits_match":
        winner, losses = model.multiple_choices_with_losses(
            image, policy["question"], policy["options"], []
        )
        record = aggregate_vstar_losses([losses])
        if record.output != winner:
            raise ValueError("root winner disagrees with option losses")
        return record
    if policy["answer_type"] == "option_list":
        requests = [
            (
                f"{policy['question']}\n{option_block}Answer the option letter directly.",
                [],
            )
            for option_block in policy["options"]
        ]
        raw = model.free_form_batch(image, requests)
        return aggregate_hr_answers(policy["options"], raw)
    raise ValueError("root_search_fallback currently supports only V* and HR-Bench answer types")


def _node_boxes(nodes: Sequence[Any]) -> tuple[tuple[int | float, ...], ...]:
    boxes: list[tuple[int | float, ...]] = []
    for node in nodes:
        try:
            box = tuple(node.state.bbox)
        except (AttributeError, TypeError):
            continue
        if len(box) == 4:
            boxes.append(box)
    return tuple(boxes)


def _ranker(config: Mapping[str, Any], scorer: Any, node_ranker: Any) -> Any:
    if not config["rerank_enabled"]:
        if node_ranker is not None or scorer is not None:
            raise ValueError("ranker/scorer supplied while rerank_enabled is false")
        return None
    if node_ranker is not None:
        if scorer is not None:
            raise ValueError("supply scorer or node_ranker, not both")
        return node_ranker
    if scorer is None:
        raise ValueError("rerank_enabled requires an injected scorer or node_ranker")
    return QueryAwareNodeRanker(
        scorer, beta=config["beta"], alpha=config["alpha"], visual_lambda=config["visual_lambda"]
    )


def get_evidence_gap_response(
    *,
    sam_model: Any,
    zoom_model: Any,
    nlp_model: Any,
    policy_annotation: Mapping[str, Any],
    original_annotation: Mapping[str, Any],
    ic_examples: Any,
    decomposed_question_template: str,
    config: str | os.PathLike[str] | Mapping[str, Any],
    image_folder: str | None = None,
    cvsearch_fn: Any = None,
    planner: Any = build_query_plan,
    policy_callback: Any = None,
    scorer: Any = None,
    node_ranker: Any = None,
    targets: Sequence[str] = (),
    answer_record_factory: Any = None,
) -> tuple[Any, MethodTrace]:
    """Run the honest minimal method without reading evaluator-only annotation fields."""
    # ``original_annotation`` is intentionally untouched; restoration is a separate boundary.
    policy = _policy_copy(policy_annotation)
    method_config = load_method_config(config)
    if not callable(planner):
        raise TypeError("planner must be callable")
    query_plan = planner(_policy_copy(policy), copy.deepcopy(targets))
    if not isinstance(query_plan, QueryPlan):
        raise TypeError("planner must return QueryPlan")
    _strict_json(query_plan.to_dict(), "query plan")
    if policy_callback is not None:
        if not callable(policy_callback):
            raise TypeError("policy_callback must be callable")
        callback_result = policy_callback(_policy_copy(policy))
        if callback_result is not None:
            _strict_json(callback_result, "policy callback result")

    ledger = BudgetLedger(
        max_mllm_calls=method_config["max_mllm_calls"],
        max_processed_pixels=method_config["max_processed_pixels"],
    )
    answer_reserve_calls = 0
    if method_config["mode"] == "rerank_only":
        if policy["answer_type"] == "logits_match":
            answer_reserve_calls = _BudgetedZoomModel._choice_calls(policy["options"])
        elif policy["answer_type"] == "option_list":
            if not isinstance(policy["options"], list) or len(policy["options"]) != 4:
                raise ValueError("HR rerank_only requires exactly four option blocks")
            answer_reserve_calls = 4
    budgeted_model = _BudgetedZoomModel(
        zoom_model,
        ledger,
        answer_reserve_calls=answer_reserve_calls,
        answer_image_loader=lambda: _image_for(policy, image_folder),
        answer_type=policy["answer_type"],
    )
    ranker = _ranker(method_config, scorer, node_ranker)
    effective_ranking_query = "cvsearch_default_order"
    if method_config["rerank_enabled"]:
        effective_ranking_query = (
            "main_query_plus_planned_augmented_queries"
            if query_plan.augmented_queries else "main_query_plus_current_visual_cue"
        )
    trace = MethodTrace(
        query_plan=copy.deepcopy(query_plan),
        method_mode=method_config["mode"],
        config_id=method_config["config_id"],
        effective_config=copy.deepcopy(method_config),
        effective_ranking_query=effective_ranking_query,
        pixel_accounting=method_config["pixel_accounting"],
    )
    runtime_annotation = _policy_copy(policy)
    observations: list[tuple[str, tuple[Any, ...], AnswerRecord, Any]] = []

    def observe(phase: str, nodes: list[Any], raw_answer: Any) -> None:
        if phase not in {"quick", "search"}:
            raise ValueError(f"unknown answer observation phase: {phase}")
        raw_snapshot = copy.deepcopy(raw_answer)
        if answer_record_factory is None:
            record = _as_answer_record(_policy_copy(policy), raw_snapshot)
        else:
            record = answer_record_factory(_policy_copy(policy), phase, raw_snapshot)
            if not isinstance(record, AnswerRecord):
                raise TypeError("answer_record_factory must return AnswerRecord")
            record = copy.deepcopy(record)
        record.selected_from = "root" if phase == "quick" else "search"
        _strict_json(record.to_dict(), "answer observation")
        observations.append((phase, tuple(nodes), record, raw_snapshot))

    if cvsearch_fn is None:
        try:
            from CVSearch import get_cvsearch_response as cvsearch_fn
        except ImportError as error:
            raise ImportError("pass cvsearch_fn or import with the cvsearch directory on sys.path") from error

    started = time.perf_counter()
    root_record: AnswerRecord | None = None
    root_cost = 0
    budget_interrupted = False
    interrupted_record: AnswerRecord | None = None
    if method_config["mode"] == "root_search_fallback":
        root_record = _root_answer(_policy_copy(policy), budgeted_model, image_folder)
        root_record.selected_from = "root"
        _strict_json(root_record.to_dict(), "root answer")
        root_cost = ledger.mllm_calls

    cvsearch_kwargs = {
        "sam_model": sam_model,
        "zoom_model": budgeted_model,
        "nlp_model": nlp_model,
        "annotation": runtime_annotation,
        "ic_examples": copy.deepcopy(ic_examples),
        "decomposed_question_template": decomposed_question_template,
        "answering_confidence_threshold_upper": 0.9,
        "answering_confidence_threshold_lower": 0.0,
        "fast_threshold": method_config["quick_gate"],
        "pop_limit": lambda max_depth: max_depth * 3,
        "threshold_descrease": [0.05, 0.1, 0.2],
        "image_folder": image_folder,
        "search_mode": True,
        "enable_parent_verification": False,
        "node_ranker": ranker,
        "answer_observer": observe,
        "method_trace": trace,
    }
    try:
        raw_response = cvsearch_fn(**cvsearch_kwargs)
    except BudgetExceeded:
        budget_interrupted = True
        if observations:
            _, _, interrupted_record, interrupted_raw = observations[-1]
            interrupted_record = copy.deepcopy(interrupted_record)
            interrupted_record.output = copy.deepcopy(interrupted_raw)
            raw_response = copy.deepcopy(interrupted_raw)
        elif root_record is not None:
            interrupted_record = copy.deepcopy(root_record)
            raw_response = copy.deepcopy(root_record.output)
        elif budgeted_model.answer_reserve_available:
            interrupted_record = _root_answer(_policy_copy(policy), budgeted_model, image_folder)
            interrupted_record.selected_from = "root"
            _strict_json(interrupted_record.to_dict(), "reserved root answer")
            raw_response = copy.deepcopy(interrupted_record.output)
        else:
            raise

    runtime_targets = runtime_annotation.get("targets")
    if not targets and planner is build_query_plan and isinstance(runtime_targets, (list, tuple)):
        runtime_plan = build_query_plan(_policy_copy(policy), runtime_targets)
        runtime_plan.evidence_items = tuple(runtime_plan.evidence_items) + ({
            "kind": "runtime_ranking_context",
            "query_source": "main_query_plus_current_visual_cue",
            "planned_augmented_queries_used": False,
        },)
        trace.query_plan = runtime_plan

    final_record: AnswerRecord
    output: Any
    if method_config["mode"] == "rerank_only":
        output = copy.deepcopy(raw_response)
        if interrupted_record is not None:
            final_record = copy.deepcopy(interrupted_record)
        else:
            final_record = copy.deepcopy(observations[-1][2]) if observations else _as_answer_record(policy, output)
        final_record.output = copy.deepcopy(output)
        if not final_record.selected_from:
            final_record.selected_from = "response"
        if budget_interrupted:
            trace.history.append(HistoryRecord(
                step=0, answer=copy.deepcopy(final_record), cost=ledger.mllm_calls
            ))
    else:
        search_records = [item for item in observations if item[0] == "search"]
        selected_search_raw: Any = None
        if budget_interrupted:
            search_record = copy.deepcopy(interrupted_record)
            if search_record is not None and search_record.selected_from != "search":
                search_record = None
        elif search_records and policy["answer_type"] == "logits_match":
            _, nodes, _, raw_observed = search_records[-1]
            image = _image_for(policy, image_folder)
            try:
                winner, losses = budgeted_model.multiple_choices_with_losses(
                    image, policy["question"], policy["options"], list(nodes)
                )
            except BudgetExceeded:
                budget_interrupted = True
                search_record = None
            else:
                if winner != raw_observed:
                    raise ValueError("searched-node loss winner disagrees with CVSearch output")
                search_record = aggregate_vstar_losses([losses])
                search_record.selected_from = "search"
        elif search_records:
            _, _, observed_record, selected_search_raw = search_records[-1]
            search_record = copy.deepcopy(observed_record)
            search_record.selected_from = "search"
        else:
            search_record = None
        if root_record is None:
            raise AssertionError("root fallback mode requires a root answer")
        if budget_interrupted and interrupted_record is not None:
            final_record = copy.deepcopy(interrupted_record)
        elif search_record is None:
            final_record = copy.deepcopy(root_record)
        elif policy["answer_type"] == "option_list" and (
            root_record.aggregation_available is False
            or search_record.aggregation_available is False
        ):
            final_record = copy.deepcopy(search_record)
            final_record.selected_from = "search"
            final_record.output = copy.deepcopy(selected_search_raw)
        else:
            final_record = select_root_or_search(
                root_record, search_record, method_config["root_fallback_tolerance"]
            )
        output = copy.deepcopy(final_record.output)
        trace.history.append(HistoryRecord(step=0, answer=copy.deepcopy(root_record), cost=root_cost))
        if search_record is not None and (
            not budget_interrupted or final_record.selected_from == "search"
        ):
            trace.history.append(HistoryRecord(step=1, answer=copy.deepcopy(search_record), cost=ledger.mllm_calls))
        elif budget_interrupted and interrupted_record is not None and interrupted_record is not root_record:
            if not _outputs_agree(interrupted_record.output, root_record.output):
                trace.history.append(HistoryRecord(
                    step=1, answer=copy.deepcopy(interrupted_record), cost=ledger.mllm_calls
                ))

    boxes = runtime_annotation.get("searched_bbox", ())
    try:
        final_boxes = tuple(tuple(box) for box in boxes)
    except TypeError as error:
        raise ValueError("searched_bbox must be a sequence of boxes") from error
    if not final_boxes and final_record.selected_from in {"root", "search", "response"}:
        matching = [item for item in observations if item[2].selected_from == final_record.selected_from]
        if matching:
            final_boxes = _node_boxes(matching[-1][1])
    if final_record.selected_from == "root":
        final_boxes = ()

    trace.final_answer = copy.deepcopy(final_record)
    trace.final_boxes = final_boxes
    trace.budget = copy.deepcopy(ledger)
    trace.elapsed_seconds = time.perf_counter() - started
    trace.termination = FORCED_RETURN
    trace.budget_interrupted = budget_interrupted
    _capture_runtime_diagnostics(trace, runtime_annotation)
    trace.steps.append(StepTrace(
        step=0,
        action=FORCED_RETURN,
        no_op_reason="minimal_v1 has no certified-stop controller",
        answer=copy.deepcopy(final_record),
        budget=copy.deepcopy(ledger),
    ))
    if not _outputs_agree(output, trace.final_answer.output):
        raise ValueError("emitted output and trace.final_answer.output disagree")
    _strict_json(trace.to_dict(), "method trace")
    return output, trace


def compose_output_record(original_annotation: Mapping[str, Any], output: Any,
                          trace: MethodTrace) -> dict[str, Any]:
    """Restore evaluator fields only after policy/model work has completed."""
    if not isinstance(original_annotation, Mapping):
        raise TypeError("original_annotation must be a mapping")
    if not isinstance(trace, MethodTrace):
        raise TypeError("trace must be MethodTrace")
    if trace.final_answer is not None and not _outputs_agree(output, trace.final_answer.output):
        raise ValueError("emitted output and trace.final_answer.output disagree")
    if "method_trace" in original_annotation or "_eg_ordinal" in original_annotation or "_eg_run_fingerprint" in original_annotation:
        raise ValueError("original annotation contains a reserved evidence-gap field")
    record = copy.deepcopy(dict(original_annotation))
    record["output"] = copy.deepcopy(output)
    trace_payload = trace.to_dict()
    record["method_trace"] = trace_payload
    if trace.cvsearch_search_mode is not None:
        record["search_mode"] = trace_payload["cvsearch_search_mode"]
    if trace.root_ans_conf is not None:
        record["root_ans_conf"] = trace_payload["root_ans_conf"]
    for key in ("num_pop", "num_zoom_in", "num_zoom_out"):
        record[key] = copy.deepcopy(trace_payload[key])
    _strict_json(record, "output record")
    return record
