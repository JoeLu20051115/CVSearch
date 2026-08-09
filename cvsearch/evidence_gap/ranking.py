"""Pure query-aware candidate ranking utilities."""

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .types import CandidateScore


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def percentile(values: Sequence[float]) -> list[float]:
    """Return ascending average-rank percentiles; constant inputs are neutral."""
    numbers = [_finite_number(value, "score") for value in values]
    count = len(numbers)
    if count == 0:
        return []
    if count == 1 or min(numbers) == max(numbers):
        return [0.5] * count

    order = sorted(range(count), key=numbers.__getitem__)
    ranks = [0.0] * count
    position = 0
    while position < count:
        end = position + 1
        while end < count and numbers[order[end]] == numbers[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        for sorted_index in range(position, end):
            ranks[order[sorted_index]] = average_rank / (count - 1)
        position = end
    return ranks


def edge_density(image: Image.Image) -> float:
    """Mean horizontal/vertical grayscale finite difference, normalized by 255."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    if grayscale.size == 0:
        return 0.0
    total = 0
    count = 0
    if grayscale.shape[1] > 1:
        differences = np.subtract(grayscale[:, 1:], grayscale[:, :-1], dtype=np.int16)
        np.abs(differences, out=differences)
        total += int(differences.sum(dtype=np.int64))
        count += differences.size
        del differences
    if grayscale.shape[0] > 1:
        differences = np.subtract(grayscale[1:, :], grayscale[:-1, :], dtype=np.int16)
        np.abs(differences, out=differences)
        total += int(differences.sum(dtype=np.int64))
        count += differences.size
        del differences
    if count == 0:
        return 0.0
    density = total / (count * 255.0)
    return min(1.0, max(0.0, density))


def _weight(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def fuse_scores(
    main: Sequence[float],
    augmented: Sequence[float],
    complexity: Sequence[float],
    edge: Sequence[float],
    beta: float,
    alpha: float,
    visual_lambda: float,
) -> list[CandidateScore]:
    """Percentile-normalize four components then apply the design's fusion formula."""
    lengths = {len(main), len(augmented), len(complexity), len(edge)}
    if len(lengths) != 1:
        raise ValueError("score components must have identical lengths")
    beta = _weight(beta, "beta")
    alpha = _weight(alpha, "alpha")
    visual_lambda = _weight(visual_lambda, "visual_lambda")
    mains, augmenteds, complexities, edges = map(percentile, (main, augmented, complexity, edge))
    scores = []
    for main_score, augmented_score, complexity_score, edge_score in zip(mains, augmenteds, complexities, edges):
        relevance = beta * main_score + (1.0 - beta) * augmented_score
        visual = visual_lambda * complexity_score + (1.0 - visual_lambda) * edge_score
        rank = alpha * relevance + (1.0 - alpha) * visual
        scores.append(CandidateScore(
            main=main_score,
            augmented=augmented_score,
            complexity=complexity_score,
            edge_density=edge_score,
            relevance=relevance,
            visual=visual,
            rank=rank,
        ))
    return scores


class QueryAwareNodeRanker:
    """Rank an existing candidate pool without pruning any node."""

    def __init__(self, scorer: Any, beta: float = 0.5, alpha: float = 0.5, visual_lambda: float = 0.5):
        if not callable(getattr(scorer, "score", None)):
            raise TypeError("scorer must provide score(images, texts)")
        self.scorer = scorer
        self.beta = _weight(beta, "beta")
        self.alpha = _weight(alpha, "alpha")
        self.visual_lambda = _weight(visual_lambda, "visual_lambda")

    @staticmethod
    def _crop(image: Image.Image, node: Any) -> tuple[Image.Image, tuple[float, float, float, float]]:
        if not isinstance(image, Image.Image):
            raise TypeError("image_pil must be a PIL image")
        try:
            bbox = tuple(node.state.bbox)
        except AttributeError as error:
            raise TypeError("node must provide state.bbox in xywh form") from error
        if len(bbox) != 4:
            raise ValueError("node bbox must have four xywh values")
        x, y, width, height = (_finite_number(value, "bbox") for value in bbox)
        if width <= 0 or height <= 0:
            raise ValueError("node bbox width and height must be positive")
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(image.width, math.ceil(x + width)), min(image.height, math.ceil(y + height))
        if right <= left or bottom <= top:
            raise ValueError("node bbox does not intersect the image")
        return image.crop((left, top, right, bottom)), (x, y, width, height)

    @staticmethod
    def _matrix(matrix: Any, rows: int, columns: int) -> list[list[float]]:
        try:
            result = [list(row) for row in matrix]
        except TypeError as error:
            raise TypeError("scorer result must be a matrix") from error
        if len(result) != rows or any(len(row) != columns for row in result):
            raise ValueError(f"scorer result must have shape {rows}x{columns}")
        return [[_finite_number(value, "scorer result") for value in row] for row in result]

    @staticmethod
    def _query(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        return value

    def rank(self, nodes: Sequence[Any], image_pil: Image.Image, main_query: str, augmented_queries: Sequence[str]) -> list[Any]:
        return self.rank_with_details(nodes, image_pil, main_query, augmented_queries)[0]

    def __call__(self, nodes: Sequence[Any], image_pil: Image.Image, main_query: str, augmented_queries: Sequence[str]) -> tuple[list[Any], list[dict[str, Any]]]:
        return self.rank_with_details(nodes, image_pil, main_query, augmented_queries)

    def rank_with_details(
        self,
        nodes: Sequence[Any],
        image_pil: Image.Image,
        main_query: str,
        augmented_queries: Sequence[str],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        candidates = list(nodes)
        if not candidates:
            return [], []
        main_query = self._query(main_query, "main_query")
        augmented_queries = () if augmented_queries is None else augmented_queries
        queries = [main_query] + [self._query(query, "augmented query") for query in augmented_queries]
        crops_and_bboxes = [self._crop(image_pil, node) for node in candidates]
        crops = [crop for crop, _ in crops_and_bboxes]
        matrix = self._matrix(self.scorer.score(crops, queries), len(candidates), len(queries))
        main = [row[0] for row in matrix]
        augmented = [row[0] if len(row) == 1 else sum(sorted(row[1:], reverse=True)[:3]) / min(3, len(row) - 1) for row in matrix]
        complexity = [_finite_number(getattr(node, "complexity", None), "node complexity") for node in candidates]
        edges = [edge_density(crop) for crop in crops]
        scores = fuse_scores(main, augmented, complexity, edges, self.beta, self.alpha, self.visual_lambda)
        indexed = list(enumerate(zip(candidates, crops_and_bboxes, scores)))
        indexed.sort(key=lambda item: item[1][2].rank, reverse=True)
        ranked = [item[1][0] for item in indexed]
        details = [
            {"node_id": getattr(node, "id", None), "bbox": list(bbox), "score": score.to_dict()}
            for _, (node, (_, bbox), score) in indexed
        ]
        return ranked, details


class ConservativeQueryRanker:
    """Blend a query ranker with CVSearch order without dropping candidates."""

    def __init__(self, base_ranker: Any, rho: float, max_displacement: int):
        if not callable(base_ranker):
            raise TypeError("base_ranker must be callable")
        self.base_ranker = base_ranker
        self.rho = _weight(rho, "rho")
        if isinstance(max_displacement, bool) or not isinstance(max_displacement, int):
            raise TypeError("max_displacement must be a non-negative integer")
        if max_displacement < 0:
            raise ValueError("max_displacement must be non-negative")
        self.max_displacement = max_displacement

    @staticmethod
    def _identity_positions(nodes: Sequence[Any], name: str) -> dict[int, int]:
        positions: dict[int, int] = {}
        for index, node in enumerate(nodes):
            identity = id(node)
            if identity in positions:
                raise ValueError(f"{name} must not contain duplicate nodes")
            positions[identity] = index
        return positions

    @staticmethod
    def _base_result(result: Any, candidates: Sequence[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("base_ranker must return (ranked_nodes, details)")
        try:
            query_ranked = list(result[0])
            raw_details = list(result[1])
        except TypeError as error:
            raise ValueError("base_ranker results must be iterable") from error
        candidate_positions = ConservativeQueryRanker._identity_positions(candidates, "candidates")
        query_positions = ConservativeQueryRanker._identity_positions(query_ranked, "base_ranker ranked nodes")
        if len(query_ranked) != len(candidates) or set(query_positions) != set(candidate_positions):
            raise ValueError("base_ranker must preserve the candidate identity multiset")
        if len(raw_details) != len(query_ranked):
            raise ValueError("base_ranker details must align with query-ranked nodes")
        details: list[dict[str, Any]] = []
        for node, detail in zip(query_ranked, raw_details):
            if not isinstance(detail, Mapping):
                raise ValueError("base_ranker details must be mappings")
            if detail.get("node_id", object()) != getattr(node, "id", None):
                raise ValueError("base_ranker details must align with query-ranked nodes")
            details.append(dict(detail))
        return query_ranked, details

    @staticmethod
    def _bounded_order(
        candidates: Sequence[Any], desired: Sequence[Any], positions: dict[int, int], max_displacement: int
    ) -> list[Any]:
        """Stable bounded insertion by each candidate's admissible output window."""
        desired_positions = {id(node): index for index, node in enumerate(desired)}
        remaining = list(candidates)
        result: list[Any] = []
        for output_position in range(len(candidates)):
            eligible = [
                node for node in remaining
                if positions[id(node)] - max_displacement <= output_position
            ]
            if not eligible:
                raise AssertionError("bounded ranking has no eligible candidate")
            due = [
                node for node in eligible
                if positions[id(node)] + max_displacement <= output_position
            ]
            pool = due if due else eligible
            selected = min(pool, key=lambda node: (
                positions[id(node)] + max_displacement if due else desired_positions[id(node)],
                positions[id(node)],
            ))
            remaining.remove(selected)
            result.append(selected)
        if any(abs(index - positions[id(node)]) > max_displacement for index, node in enumerate(result)):
            raise AssertionError("bounded ranking exceeded maximum displacement")
        return result

    def __call__(
        self,
        nodes: Sequence[Any],
        image_pil: Image.Image,
        main_query: str,
        augmented_queries: Sequence[str],
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        candidates = list(nodes)
        if not candidates:
            return [], []
        cvsearch_positions = self._identity_positions(candidates, "candidates")
        query_ranked, query_details = self._base_result(
            self.base_ranker(candidates, image_pil, main_query, augmented_queries), candidates
        )
        query_positions = self._identity_positions(query_ranked, "base_ranker ranked nodes")
        details_by_identity = {id(node): detail for node, detail in zip(query_ranked, query_details)}
        desired = sorted(
            candidates,
            key=lambda node: (
                -(
                    (1.0 - self.rho) / (60.0 + cvsearch_positions[id(node)])
                    + self.rho / (60.0 + query_positions[id(node)])
                ),
                cvsearch_positions[id(node)],
            ),
        )
        ranked = self._bounded_order(candidates, desired, cvsearch_positions, self.max_displacement)
        details: list[dict[str, Any]] = []
        for node in ranked:
            cvsearch_rank = cvsearch_positions[id(node)]
            query_rank = query_positions[id(node)]
            detail = details_by_identity[id(node)]
            enriched = dict(detail)
            enriched.update({
                "cvsearch_rank": cvsearch_rank,
                "query_rank": query_rank,
                "fused_rank_score": (
                    (1.0 - self.rho) / (60.0 + cvsearch_rank)
                    + self.rho / (60.0 + query_rank)
                ),
            })
            details.append(enriched)
        return ranked, details
