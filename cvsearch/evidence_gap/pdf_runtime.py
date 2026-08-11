"""Runtime adapters for PDF-faithful query planning, trees, and uncertainty."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
from PIL import Image

from cvsearch.eval.phase7_uncertainty_confirmation import confirmation_prompt_material
from cvsearch.evidence_gap.answers import (
    aggregate_hr_answers,
    aggregate_vstar_losses,
    canonical_text,
    official_letter,
    parse_option_block,
)
from cvsearch.evidence_gap.input import sanitize_annotation
from cvsearch.evidence_gap.method import build_query_plan
from cvsearch.evidence_gap.search_state import SearchStateCollector
from cvsearch.evidence_gap.types import AnswerRecord, sanitize_evidence_requirements
from cvsearch.models.tree import NodeA, NodeState

from .pdf_controller import ActionOutcome, FrozenRankedQueue
from .pdf_types import ActionName, CandidateDescriptor, SearchStateRecord


_PLAN_PROMPT = (
    "Create answer-free visual search material for the question below. Return only one "
    "JSON object with exactly these keys: augmented_queries, evidence_items, "
    "global_scope_required. augmented_queries must contain four to six distinct short "
    "phrases for locating visible objects, details, or relations. evidence_items must use "
    "the target_detail, relation_context, coverage, or question_evidence schemas. Do not "
    "answer the question, mention candidate answers, or infer unseen details.\nQuestion: {question}"
)


def _normalized_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not (result := " ".join(value.split())):
        raise ValueError(f"{name} must be nonempty text")
    return result


def _strict_json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be strict JSON") from error


@dataclass(frozen=True)
class PDFQueryPlan:
    main_query: str
    targets: tuple[str, ...]
    augmented_queries: tuple[str, ...]
    evidence_items: tuple[dict[str, Any], ...]
    global_scope_required: bool
    fallback_used: bool
    fallback_reason: str | None
    raw_response_sha256: str

    def __post_init__(self) -> None:
        _normalized_text(self.main_query, "main_query")
        for name, values in (("targets", self.targets), ("augmented_queries", self.augmented_queries)):
            if not isinstance(values, tuple) or not all(isinstance(item, str) and item for item in values):
                raise TypeError(f"{name} must be an immutable text tuple")
            if len({item.casefold() for item in values}) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if len(self.augmented_queries) < 3:
            raise ValueError("augmented_queries must retain at least three queries")
        if not isinstance(self.evidence_items, tuple) or not self.evidence_items:
            raise ValueError("evidence_items must be a nonempty immutable tuple")
        sanitize_evidence_requirements(self.evidence_items)
        if type(self.global_scope_required) is not bool or type(self.fallback_used) is not bool:
            raise TypeError("query-plan flags must be booleans")
        if self.fallback_used != (self.fallback_reason is not None):
            raise ValueError("fallback reason must exist exactly when fallback is used")
        if self.fallback_reason is not None:
            _normalized_text(self.fallback_reason, "fallback_reason")
        if not re.fullmatch(r"[0-9a-f]{64}", self.raw_response_sha256):
            raise ValueError("raw_response_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "main_query": self.main_query,
            "targets": list(self.targets),
            "augmented_queries": list(self.augmented_queries),
            "evidence_items": _strict_json_copy(self.evidence_items, "evidence_items"),
            "global_scope_required": self.global_scope_required,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "raw_response_sha256": self.raw_response_sha256,
        }


def _option_texts(policy: Mapping[str, Any]) -> tuple[str, ...]:
    options = policy["options"]
    if not isinstance(options, list):
        raise TypeError("options must be a list")
    if policy["answer_type"] == "logits_match":
        return tuple(_normalized_text(item, "option") for item in options)
    if policy["answer_type"] == "option_list":
        texts = []
        for block in options:
            texts.extend(parse_option_block(block).values())
        return tuple(dict.fromkeys(texts))
    return tuple(_normalized_text(item, "option") for item in options)


def _parse_plan(raw: str, policy: Mapping[str, Any], targets: tuple[str, ...]) -> PDFQueryPlan:
    if not isinstance(raw, str):
        raise TypeError("structured query plan response must be text")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("structured query plan response contains no JSON object")
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict) or set(data) != {
        "augmented_queries", "evidence_items", "global_scope_required",
    }:
        raise ValueError("structured query plan has an invalid exact schema")
    queries = data["augmented_queries"]
    if not isinstance(queries, list) or not 3 <= len(queries) <= 8:
        raise ValueError("structured query plan needs three to eight augmentations")
    normalized_queries = tuple(_normalized_text(item, "augmented query") for item in queries)
    if len({item.casefold() for item in normalized_queries}) != len(normalized_queries):
        raise ValueError("structured query augmentations must be distinct")
    if not isinstance(data["evidence_items"], list) or not data["evidence_items"]:
        raise ValueError("structured query plan needs evidence items")
    evidence_items = tuple(_strict_json_copy(item, "evidence item") for item in data["evidence_items"])
    sanitize_evidence_requirements(evidence_items)
    if type(data["global_scope_required"]) is not bool:
        raise TypeError("global_scope_required must be a boolean")

    material = json.dumps(
        {"augmented_queries": normalized_queries, "evidence_items": evidence_items},
        ensure_ascii=False,
    ).casefold()
    for option in _option_texts(policy):
        normalized = option.casefold()
        if normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", material):
            raise ValueError("structured query plan leaked answer-option text")
    return PDFQueryPlan(
        main_query=policy["question"],
        targets=targets,
        augmented_queries=normalized_queries,
        evidence_items=evidence_items,
        global_scope_required=data["global_scope_required"],
        fallback_used=False,
        fallback_reason=None,
        raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


def _fallback_plan(
    policy: Mapping[str, Any], targets: tuple[str, ...], raw: str, reason: str,
) -> PDFQueryPlan:
    base = build_query_plan(policy, targets)
    subjects = list(targets)
    if not subjects:
        content = re.findall(r"[A-Za-z0-9]+", policy["question"])
        subjects = [" ".join(content[-6:]) or "question evidence"]
    queries = []
    for subject in subjects:
        queries.extend((
            f"locate {subject}",
            f"inspect visible details of {subject}",
            f"surrounding context and relations of {subject}",
        ))
    normalized = tuple(dict.fromkeys(" ".join(item.split()) for item in queries))
    while len(normalized) < 3:
        normalized += (f"visible evidence view {len(normalized) + 1}",)
    return PDFQueryPlan(
        main_query=policy["question"],
        targets=targets,
        augmented_queries=normalized,
        evidence_items=tuple(_strict_json_copy(item, "evidence item") for item in base.evidence_items),
        global_scope_required=base.global_scope_required,
        fallback_used=True,
        fallback_reason=reason,
        raw_response_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


def build_pdf_query_plan(
    policy_annotation: Mapping[str, Any],
    targets: Sequence[str],
    *,
    generator: Callable[[str], str],
) -> PDFQueryPlan:
    policy = sanitize_annotation(policy_annotation)
    if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
        raise TypeError("targets must be a sequence")
    normalized_targets = tuple(dict.fromkeys(
        _normalized_text(item, "target") for item in targets
    ))
    if not callable(generator):
        raise TypeError("generator must be callable")
    prompt = _PLAN_PROMPT.format(question=policy["question"])
    raw = generator(prompt)
    if not isinstance(raw, str):
        raise TypeError("generator must return text")
    try:
        return _parse_plan(raw, policy, normalized_targets)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return _fallback_plan(policy, normalized_targets, raw, type(error).__name__)


@dataclass(frozen=True)
class CatalogNode:
    descriptor: CandidateDescriptor
    parent_key: str | None
    child_keys: tuple[str, ...]
    source: str | None
    complexity: float
    synthetic: bool = False


class TreeCatalog:
    """Immutable parent/child catalog backed by SearchStateCollector renderers."""

    def __init__(
        self,
        nodes: Mapping[str, CatalogNode],
        root_key: str,
        collector: SearchStateCollector,
        image: Image.Image,
    ) -> None:
        self._nodes = dict(nodes)
        self.root_key = root_key
        self._collector = collector
        self._image = image.copy()
        if root_key not in self._nodes:
            raise ValueError("tree root is missing from catalog")
        for key in self._nodes:
            self.path_to(key)

    @classmethod
    def from_collector(cls, collector: SearchStateCollector, image: Image.Image) -> "TreeCatalog":
        if not isinstance(collector, SearchStateCollector):
            raise TypeError("collector must be SearchStateCollector")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be PIL.Image")
        payload = collector.to_dict()
        snapshots = payload["snapshots"]
        tree_snapshots = [item for item in snapshots if item["event"] == "tree_ready"]
        source_snapshots = tree_snapshots or [
            item for item in snapshots if item["event"] == "p0_selected"
        ]
        if not source_snapshots:
            raise ValueError("collector contains neither a full tree nor P0 candidates")

        raw_nodes: dict[str, dict[str, Any]] = {}
        group_by_key: dict[str, str] = {}
        for snapshot in source_snapshots:
            group_prefix = f"tree-{snapshot['search_call_ordinal']}"
            for index, raw in enumerate(snapshot["candidates"]):
                key = raw["canonical_key"]
                raw_nodes.setdefault(key, raw)
                group_by_key.setdefault(
                    key, raw.get("parent_key") or f"{group_prefix}-root",
                )

        full_root = next((
            raw for raw in raw_nodes.values()
            if raw.get("depth") == 0
            and tuple(raw.get("bbox_original", ())) == (0, 0, image.width, image.height)
        ), None)
        synthetic_key = "pdf-root-" + hashlib.sha256(
            json.dumps({
                "mode": image.mode, "size": [image.width, image.height],
                "pixels": hashlib.sha256(image.tobytes()).hexdigest(),
            }, sort_keys=True).encode()
        ).hexdigest()
        root_key = full_root["canonical_key"] if full_root is not None else synthetic_key
        nodes: dict[str, CatalogNode] = {}
        for native_ordinal, (key, raw) in enumerate(raw_nodes.items()):
            parent = raw.get("parent_key")
            if full_root is None and parent is None:
                parent = root_key
            children = tuple(raw.get("child_keys") or ())
            complexity = raw.get("complexity")
            if complexity is None:
                complexity = raw.get("prior_prob", 0.0)
            complexity = float(complexity)
            if not math.isfinite(complexity):
                raise ValueError("tree complexity must be finite")
            descriptor = CandidateDescriptor(
                canonical_key=key,
                sibling_group=parent or group_by_key[key],
                native_ordinal=native_ordinal,
                bbox_original=tuple(raw["bbox_original"]),
                depth=int(raw["depth"]),
                render_level=int(raw["render_level"]),
            )
            nodes[key] = CatalogNode(
                descriptor, parent, children, raw.get("source"), complexity,
            )
        if full_root is None:
            top_level = tuple(key for key, node in nodes.items() if node.parent_key == root_key)
            descriptor = CandidateDescriptor(
                canonical_key=root_key,
                sibling_group="synthetic-root",
                native_ordinal=0,
                bbox_original=(0.0, 0.0, float(image.width), float(image.height)),
                depth=0,
                render_level=0,
            )
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            complexity = float(pixels.var() / (255.0 ** 2))
            nodes[root_key] = CatalogNode(
                descriptor, None, top_level, "global", complexity, True,
            )
        for key, node in nodes.items():
            if node.parent_key is not None and node.parent_key not in nodes:
                raise ValueError(f"tree node {key} has a missing parent")
            if any(child not in nodes for child in node.child_keys):
                raise ValueError(f"tree node {key} has a missing child")
        return cls(nodes, root_key, collector, image)

    def node(self, key: str) -> CatalogNode:
        try:
            return self._nodes[key]
        except KeyError as error:
            raise KeyError(f"unknown tree node: {key}") from error

    def children(self, key: str) -> tuple[str, ...]:
        return self.node(key).child_keys

    def path_to(self, key: str) -> tuple[str, ...]:
        path = []
        seen = set()
        current: str | None = key
        while current is not None:
            if current in seen:
                raise ValueError("tree catalog contains a parent cycle")
            seen.add(current)
            path.append(current)
            current = self.node(current).parent_key
        path.reverse()
        if not path or path[0] != self.root_key:
            raise ValueError("tree path does not begin at the catalog root")
        return tuple(path)

    def render_nodes(self, keys: Sequence[str]) -> list[NodeA]:
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
            raise TypeError("render keys must be a sequence")
        result = []
        for key in keys:
            item = self.node(key)
            if item.synthetic:
                node = NodeA(NodeState(self._image.copy(), list(item.descriptor.bbox_original)))
                node.search_source = "global"
            else:
                view = self._collector.support_view((key,))
                if view is None or len(view) != 1:
                    raise ValueError("tree node has no native render descriptor")
                node = view[0].render_node
            node.depth = item.descriptor.depth
            node.complexity = item.complexity
            node.is_root = key == self.root_key
            node.is_leaf = not item.child_keys
            result.append(node)
        return result


class TreeActionAdapter:
    """Bind the pure controller actions to a jointly ranked CVSearch tree."""

    def __init__(
        self,
        catalog: TreeCatalog,
        image: Image.Image,
        query_plan: PDFQueryPlan,
        ranker: Any,
        *,
        max_zoom_level: int = 2,
    ) -> None:
        if not isinstance(catalog, TreeCatalog):
            raise TypeError("catalog must be TreeCatalog")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be PIL.Image")
        if not isinstance(query_plan, PDFQueryPlan):
            raise TypeError("query_plan must be PDFQueryPlan")
        if not callable(ranker):
            raise TypeError("ranker must be callable")
        if isinstance(max_zoom_level, bool) or not isinstance(max_zoom_level, int) or max_zoom_level < 1:
            raise ValueError("max_zoom_level must be a positive integer")
        self.catalog = catalog
        self.image = image.copy()
        self.query_plan = query_plan
        self.ranker = ranker
        self.max_zoom_level = max_zoom_level
        self.queue = FrozenRankedQueue()
        self.ranking_details: list[dict[str, Any]] = []
        self._ranked_children: dict[str, tuple[str, ...]] = {}
        self.reveal_children(catalog.root_key)

    def reveal_children(self, parent_key: str) -> tuple[str, ...]:
        if parent_key in self._ranked_children:
            return self._ranked_children[parent_key]
        child_keys = self.catalog.children(parent_key)
        if not child_keys:
            self._ranked_children[parent_key] = ()
            return ()
        nodes = self.catalog.render_nodes(child_keys)
        result = self.ranker(
            nodes, self.image, self.query_plan.main_query,
            self.query_plan.augmented_queries,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("tree ranker must return ranked nodes and details")
        ranked_nodes, details = list(result[0]), list(result[1])
        if len(ranked_nodes) != len(nodes) or len(details) != len(nodes):
            raise ValueError("tree ranker must preserve every child")
        key_by_identity = {id(node): key for node, key in zip(nodes, child_keys)}
        if set(map(id, ranked_nodes)) != set(key_by_identity):
            raise ValueError("tree ranker changed the child identity set")
        ranked_keys = tuple(key_by_identity[id(node)] for node in ranked_nodes)
        score_by_key: dict[str, float] = {}
        enriched = []
        for key, detail in zip(ranked_keys, details):
            if not isinstance(detail, Mapping):
                raise TypeError("tree rank details must be mappings")
            score = detail.get("score")
            if not isinstance(score, Mapping) or "rank" not in score:
                raise ValueError("tree rank detail must contain the combined rank")
            rank = float(score["rank"])
            if not math.isfinite(rank) or not 0.0 <= rank <= 1.0:
                raise ValueError("tree combined rank must be finite in [0, 1]")
            score_by_key[key] = rank
            item = _strict_json_copy(dict(detail), "tree rank detail")
            item.update({"canonical_key": key, "parent_key": parent_key})
            enriched.append(item)
        self.queue.add_sibling_group(
            [self.catalog.node(key).descriptor for key in child_keys],
            ranked_keys, score_by_key,
        )
        self.ranking_details.extend(enriched)
        self._ranked_children[parent_key] = ranked_keys
        return ranked_keys

    @staticmethod
    def _visited(state: SearchStateRecord) -> set[str]:
        return set(state.visited_keys)

    def _zoom_level(self, state: SearchStateRecord) -> int:
        prefix = f"{state.path_keys[-1]}@zoom"
        levels = [
            int(item[len(prefix):]) for item in state.observation_keys
            if item.startswith(prefix) and item[len(prefix):].isdigit()
        ]
        return max(levels, default=0)

    @staticmethod
    def _outside_area(
        candidate: tuple[float, float, float, float],
        focus: tuple[float, float, float, float],
    ) -> float:
        cx, cy, cw, ch = candidate
        fx, fy, fw, fh = focus
        overlap_w = max(0.0, min(cx + cw, fx + fw) - max(cx, fx))
        overlap_h = max(0.0, min(cy + ch, fy + fh) - max(cy, fy))
        return cw * ch - overlap_w * overlap_h

    def _expand_candidate(self, state: SearchStateRecord) -> CandidateDescriptor | None:
        focus = self.catalog.node(state.path_keys[-1]).descriptor.bbox_original
        excluded = set(state.visited_keys) | set(state.context_keys) | {self.catalog.root_key}
        for candidate in self.queue.ranked_remaining(tuple(excluded)):
            if self._outside_area(candidate.bbox_original, focus) > 0.0:
                return candidate
        return None

    def _next_candidate(self, state: SearchStateRecord) -> CandidateDescriptor | None:
        excluded = set(state.visited_keys) | {self.catalog.root_key}
        remaining = self.queue.ranked_remaining(tuple(excluded))
        return None if not remaining else remaining[0]

    def feasible(self, state: SearchStateRecord) -> dict[ActionName, bool]:
        if not isinstance(state, SearchStateRecord):
            raise TypeError("state must be SearchStateRecord")
        focus = state.path_keys[-1]
        children = self.catalog.children(focus)
        return {
            ActionName.ZOOM: self._zoom_level(state) < self.max_zoom_level,
            ActionName.SPLIT: any(key not in self._visited(state) for key in children),
            ActionName.EXPAND: self._expand_candidate(state) is not None,
            ActionName.NEXT: self._next_candidate(state) is not None,
        }

    @staticmethod
    def _unique(values: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))

    def execute(self, action: ActionName, state: SearchStateRecord) -> ActionOutcome:
        if action is ActionName.ZOOM:
            level = self._zoom_level(state) + 1
            if level > self.max_zoom_level:
                return ActionOutcome.no_op("maximum_zoom_level")
            return ActionOutcome.changed(
                path_keys=state.path_keys,
                focus_keys=state.focus_keys,
                context_keys=state.context_keys,
                visited_keys=state.visited_keys,
                observation_keys=state.observation_keys + (f"{state.path_keys[-1]}@zoom{level}",),
            )
        if action is ActionName.SPLIT:
            ranked = self.reveal_children(state.path_keys[-1])
            key = next((item for item in ranked if item not in self._visited(state)), None)
            if key is None:
                return ActionOutcome.no_op("leaf_or_children_visited")
            return ActionOutcome.changed(
                path_keys=state.path_keys + (key,),
                focus_keys=(key,),
                context_keys=state.context_keys,
                visited_keys=self._unique(state.visited_keys + (key,)),
                observation_keys=state.observation_keys + (f"{key}@base",),
            )
        if action is ActionName.EXPAND:
            candidate = self._expand_candidate(state)
            if candidate is None:
                return ActionOutcome.no_op("no_ranked_spatial_context")
            key = candidate.canonical_key
            return ActionOutcome.changed(
                path_keys=state.path_keys,
                focus_keys=state.focus_keys,
                context_keys=self._unique(state.context_keys + (key,)),
                visited_keys=self._unique(state.visited_keys + (key,)),
                observation_keys=state.observation_keys + (f"context:{key}",),
            )
        if action is ActionName.NEXT:
            candidate = self._next_candidate(state)
            if candidate is None:
                return ActionOutcome.no_op("ranked_queue_empty")
            key = candidate.canonical_key
            return ActionOutcome.changed(
                path_keys=self.catalog.path_to(key),
                focus_keys=(key,),
                context_keys=(),
                visited_keys=self._unique(state.visited_keys + (key,)),
                observation_keys=state.observation_keys + (f"{key}@base",),
            )
        raise ValueError("TreeActionAdapter executes only evidence-gap actions")

    def branch_available(self, state: SearchStateRecord) -> bool:
        visited = self._visited(state)
        return any(
            any(child not in visited for child in self.catalog.children(ancestor))
            for ancestor in state.path_keys
        )

    def render_state(self, state: SearchStateRecord) -> tuple[Image.Image, list[NodeA]]:
        level = self._zoom_level(state)
        if level:
            x, y, width, height = self.catalog.node(
                state.path_keys[-1]
            ).descriptor.bbox_original
            crop = self.image.crop((x, y, x + width, y + height))
            scale = 2 ** level
            size = (
                min(4096, max(1, round(crop.width * scale))),
                min(4096, max(1, round(crop.height * scale))),
            )
            return crop.resize(size, Image.Resampling.BICUBIC), []
        keys = self._unique(state.focus_keys + state.context_keys)
        return self.image.copy(), self.catalog.render_nodes(keys)


def _combined_uncertainty(
    record: AnswerRecord,
    prompt_winners: Sequence[Any],
    *,
    agreement: float | None = None,
) -> AnswerRecord:
    winners = list(prompt_winners)
    counts: dict[Any, int] = {}
    for winner in winners:
        counts[winner] = counts.get(winner, 0) + 1
    frequency = (
        max(counts.values()) / len(winners) if agreement is None and winners
        else record.frequency if agreement is None else agreement
    )
    if record.aggregation_available is False:
        confidence = 0.0
    else:
        confidence = (record.margin + frequency) / 2.0
    groups = dict(record.groups)
    groups["prompt_winners"] = winners
    groups["uncertainty_formula"] = "1-mean(normalized_margin,prompt_agreement)"
    return AnswerRecord(
        output=record.output,
        canonical_answer=record.canonical_answer,
        raw_outputs=record.raw_outputs,
        groups=groups,
        frequency=frequency,
        margin=record.margin,
        confidence=confidence,
        uncertainty=1.0 - confidence,
        losses=record.losses,
        selected_from="pdf_state",
        aggregation_available=(
            True if record.aggregation_available is None else record.aggregation_available
        ),
        aggregation_reason=record.aggregation_reason,
    )


def answer_with_uncertainty(
    model: Any,
    policy_annotation: Mapping[str, Any],
    image: Image.Image,
    searched_nodes: Sequence[Any],
) -> AnswerRecord:
    policy = sanitize_annotation(policy_annotation)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be PIL.Image")
    nodes = list(searched_nodes)
    material = confirmation_prompt_material(
        policy["answer_type"], policy["question"], policy["options"],
    )
    if policy["answer_type"] == "logits_match":
        rows = []
        winners = []
        for prompt in material["prompts"]:
            winner, losses = model.multiple_choices_with_losses(
                image, prompt, policy["options"], nodes,
            )
            row = [float(value) for value in losses]
            rows.append(row)
            winners.append(int(winner))
        return _combined_uncertainty(aggregate_vstar_losses(rows), winners)
    if policy["answer_type"] == "option_list":
        outputs = [
            model.free_form_using_nodes(image, prompt, nodes)
            for prompt in material["prompts"]
        ]
        record = aggregate_hr_answers(policy["options"], outputs)
        semantic_winners = []
        for block, raw in zip(policy["options"], outputs):
            letter = official_letter(raw)
            options = parse_option_block(block)
            semantic_winners.append(
                None if letter not in options else canonical_text(options[letter])
            )
        return _combined_uncertainty(
            record, semantic_winners, agreement=record.frequency,
        )
    raise ValueError("PDF uncertainty supports only V* and HR-Bench answer types")
