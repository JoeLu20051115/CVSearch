"""Independent dense query-evidence candidate generation and selection."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat

from cvsearch.eval.phase7_uncertainty_confirmation import (
    CONFIRMATION_BACKGROUND_RGB,
    CONFIRMATION_SEPARATOR_PIXELS,
    _confidence,
    _exact_rgb_image,
    _exact_snapshot,
    _image_sha256,
    _p0,
    _sha256,
    _snapshot_json,
    _vstar_confirmation_record,
)
from cvsearch.evidence_gap.answers import aggregate_hr_answers


DENSE_GRIDS = (2, 3, 4)
DENSE_OVERLAP_FRACTION = 0.125
DENSE_PANEL_SIZE = 448
DENSE_BACKGROUND_RGB = CONFIRMATION_BACKGROUND_RGB
DENSE_SEPARATOR_PIXELS = CONFIRMATION_SEPARATOR_PIXELS
DENSE_BOX_RGB = (255, 0, 0)
DENSE_TOP_K = 3
DENSE_MIN_CONFIDENCES = frozenset({0.5, 2 / 3, 0.75})
DENSE_MIN_GAINS = frozenset({0.0, 0.1, 0.25})
_CANDIDATE_FIELDS = frozenset({
    "action", "feasible", "output", "stability",
    "tile_majority_fraction", "sheet_sha256", "rank_sha256",
})


@dataclass(frozen=True)
class DenseTile:
    grid: int
    row: int
    col: int
    box: tuple[int, int, int, int]

    @property
    def identity(self) -> str:
        return f"g{self.grid}-r{self.row}-c{self.col}"


@dataclass(frozen=True)
class RankedDenseTile:
    tile: DenseTile
    relevance: float
    edge_density: float
    variance: float
    relevance_percentile: float
    edge_percentile: float
    variance_percentile: float
    visual_information: float
    score: float


@dataclass(frozen=True)
class DenseDecision:
    action: str
    status: str
    output: Any
    confidence: float | None
    confidence_gain: float | None


def generate_dense_tiles(width: int, height: int) -> tuple[DenseTile, ...]:
    if (
        isinstance(width, bool) or not isinstance(width, int) or width <= 0
        or isinstance(height, bool) or not isinstance(height, int) or height <= 0
    ):
        raise ValueError("dense source dimensions must be positive integers")
    result = []
    seen = set()
    for grid in DENSE_GRIDS:
        for row in range(grid):
            base_y0 = math.floor(row * height / grid)
            base_y1 = math.ceil((row + 1) * height / grid)
            pad_y = (base_y1 - base_y0) * DENSE_OVERLAP_FRACTION
            y0 = max(0, math.floor(base_y0 - pad_y))
            y1 = min(height, math.ceil(base_y1 + pad_y))
            for col in range(grid):
                base_x0 = math.floor(col * width / grid)
                base_x1 = math.ceil((col + 1) * width / grid)
                pad_x = (base_x1 - base_x0) * DENSE_OVERLAP_FRACTION
                x0 = max(0, math.floor(base_x0 - pad_x))
                x1 = min(width, math.ceil(base_x1 + pad_x))
                box = (x0, y0, x1, y1)
                if box in seen:
                    continue
                seen.add(box)
                result.append(DenseTile(grid, row, col, box))
    return tuple(result)


def _finite_values(values: Any, count: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must align with every dense tile")
    result = []
    for value in values:
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(float(value))
    return tuple(result)


def _percentiles(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) == 1:
        return (0.5,)
    denominator = len(values) - 1
    return tuple(
        (
            sum(other < value for other in values)
            + (sum(other == value for other in values) - 1) / 2
        ) / denominator
        for value in values
    )


def rank_dense_tiles(
    tiles: Sequence[DenseTile], relevance: Any, edge_density: Any,
    variance: Any,
) -> tuple[RankedDenseTile, ...]:
    if not isinstance(tiles, (list, tuple)) or not tiles:
        raise ValueError("dense ranking requires a nonempty tile sequence")
    frozen_tiles = tuple(tiles)
    if not all(type(tile) is DenseTile for tile in frozen_tiles):
        raise TypeError("dense ranking tiles must be exact DenseTile values")
    count = len(frozen_tiles)
    relevance = _finite_values(relevance, count, "dense relevance")
    edge_density = _finite_values(edge_density, count, "dense edge density")
    variance = _finite_values(variance, count, "dense variance")
    grouped = defaultdict(list)
    for index, tile in enumerate(frozen_tiles):
        grouped[tile.grid].append(index)
    rel_pct = [0.0] * count
    edge_pct = [0.0] * count
    var_pct = [0.0] * count
    for indices in grouped.values():
        for target, values in (
            (rel_pct, relevance), (edge_pct, edge_density),
            (var_pct, variance),
        ):
            group_percentiles = _percentiles([values[index] for index in indices])
            for index, percentile in zip(indices, group_percentiles):
                target[index] = percentile
    ranked = []
    for index, tile in enumerate(frozen_tiles):
        visual = (edge_pct[index] + var_pct[index]) / 2
        score = 0.70 * rel_pct[index] + 0.30 * visual
        ranked.append(RankedDenseTile(
            tile=tile,
            relevance=relevance[index],
            edge_density=edge_density[index],
            variance=variance[index],
            relevance_percentile=rel_pct[index],
            edge_percentile=edge_pct[index],
            variance_percentile=var_pct[index],
            visual_information=visual,
            score=score,
        ))
    ranked.sort(key=lambda item: (
        -item.score,
        -item.relevance_percentile,
        1 / (item.tile.grid * item.tile.grid),
        -item.tile.grid,
        item.tile.row,
        item.tile.col,
    ))
    return tuple(ranked)


def dense_query_texts(question: str) -> tuple[str, str, str]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("dense ranking question must be a nonempty string")
    return (
        question,
        f"visual region needed to answer: {question}",
        f"visual evidence for: {question}",
    )


def dense_visual_features(
    source: Image.Image, tiles: Sequence[DenseTile],
) -> tuple[list[float], list[float]]:
    """Compute normalized edge density and variance without mutating RGB."""
    source = _exact_rgb_image(source, "dense visual source")
    if not isinstance(tiles, (list, tuple)) or not tiles:
        raise ValueError("dense visual features require nonempty tiles")
    edges = []
    variances = []
    for tile in tiles:
        if type(tile) is not DenseTile:
            raise TypeError("dense visual feature tiles must be exact DenseTile values")
        crop = source.crop(tile.box).convert("L")
        crop.thumbnail((224, 224), Image.Resampling.LANCZOS)
        variance = float(ImageStat.Stat(crop).var[0]) / 16256.25
        edge = crop.filter(ImageFilter.FIND_EDGES)
        if edge.width > 2 and edge.height > 2:
            edge = edge.crop((1, 1, edge.width - 1, edge.height - 1))
        edge_density = float(ImageStat.Stat(edge).mean[0]) / 255.0
        edges.append(min(1.0, max(0.0, edge_density)))
        variances.append(min(1.0, max(0.0, variance)))
    return edges, variances


def _letterbox(image: Image.Image) -> Image.Image:
    panel = Image.new(
        "RGB", (DENSE_PANEL_SIZE, DENSE_PANEL_SIZE), DENSE_BACKGROUND_RGB,
    )
    fitted = ImageOps.contain(
        image, panel.size, method=Image.Resampling.LANCZOS,
    )
    panel.paste(fitted, (
        (panel.width - fitted.width) // 2,
        (panel.height - fitted.height) // 2,
    ))
    return panel


def render_dense_evidence_sheet(
    source: Image.Image, tile: DenseTile,
) -> tuple[Image.Image, dict[str, Any]]:
    source = _exact_rgb_image(source, "dense source image")
    if type(tile) is not DenseTile:
        raise TypeError("dense evidence tile must be an exact DenseTile")
    x0, y0, x1, y1 = tile.box
    if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
        raise ValueError("dense evidence tile exceeds the source image")
    focus = _letterbox(source.crop(tile.box))
    marked = source.copy()
    thickness = max(2, min(source.size) // 200)
    ImageDraw.Draw(marked).rectangle(
        (x0, y0, x1 - 1, y1 - 1), outline=DENSE_BOX_RGB, width=thickness,
    )
    context = _letterbox(marked)
    focus_hash = _image_sha256(focus)
    context_hash = _image_sha256(context)
    if focus_hash == context_hash:
        raise ValueError("dense focus and context panels must be pixel-distinct")
    sheet = Image.new(
        "RGB",
        (DENSE_PANEL_SIZE, DENSE_PANEL_SIZE * 2 + DENSE_SEPARATOR_PIXELS),
        DENSE_BACKGROUND_RGB,
    )
    sheet.paste(focus, (0, 0))
    sheet.paste(context, (0, DENSE_PANEL_SIZE + DENSE_SEPARATOR_PIXELS))
    return sheet, {
        "tile_identity": tile.identity,
        "tile_box": list(tile.box),
        "focus_panel_sha256": focus_hash,
        "context_panel_sha256": context_hash,
        "sheet_sha256": _image_sha256(sheet),
        "sheet_size": [sheet.width, sheet.height],
    }


def _majority(values: Sequence[Any]) -> tuple[Any, int]:
    counts = Counter(values)
    return min(counts.items(), key=lambda item: (-item[1], str(item[0])))


def project_dense_candidate(
    answer_type: str, options: Sequence[str], tile_observations: Any,
) -> dict[str, Any]:
    if type(tile_observations) is not list or len(tile_observations) != DENSE_TOP_K:
        raise ValueError("dense projection requires exactly three tile observations")
    if answer_type == "logits_match":
        record = _vstar_confirmation_record(
            options, tile_observations, "dense V* tiles",
        )
        majority_output, majority_count = _majority([
            observation["winner"] for observation in tile_observations
        ])
        fraction = majority_count / DENSE_TOP_K
        feasible = (
            majority_count >= 2
            and majority_output == record.output
            and record.aggregation_available is not False
        )
        confidence = min(float(record.confidence), fraction)
        canonical_answer = record.output
        tile_outputs = [observation["winner"] for observation in tile_observations]
    elif answer_type == "option_list":
        records = []
        for index, observation in enumerate(tile_observations):
            if (
                type(observation) is not list or len(observation) != 4
                or not all(isinstance(item, str) for item in observation)
            ):
                raise ValueError(
                    f"dense HR tile {index} must contain four string observations"
                )
            records.append(aggregate_hr_answers(list(options), observation))
        canonical_outputs = [record.canonical_answer for record in records]
        majority_output, majority_count = _majority(canonical_outputs)
        fraction = majority_count / DENSE_TOP_K
        majority_records = [
            record for record in records
            if record.canonical_answer == majority_output
        ]
        available = all(
            record.aggregation_available is not False for record in majority_records
        )
        feasible = majority_count >= 2 and majority_output is not None and available
        confidence = min(
            fraction,
            sum(float(record.confidence) for record in majority_records)
            / len(majority_records),
        )
        canonical_answer = majority_output
        tile_outputs = canonical_outputs
        record = majority_records[0]
    else:
        raise ValueError("dense projection answer type is unsupported")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("dense candidate confidence must be finite in [0, 1]")
    return {
        "feasible": feasible,
        "output": _snapshot_json(record.output) if feasible else None,
        "confidence": confidence,
        "canonical_answer": _snapshot_json(canonical_answer),
        "tile_majority_fraction": fraction,
        "tile_outputs": _snapshot_json(tile_outputs),
    }


def _candidate(value: Any) -> tuple[dict[str, Any], float | None]:
    candidate = _exact_snapshot(value, _CANDIDATE_FIELDS, "dense candidate")
    if candidate["action"] != "DENSE" or type(candidate["feasible"]) is not bool:
        raise ValueError("dense candidate action or feasibility is invalid")
    hashes = candidate["sheet_sha256"]
    if type(hashes) is not list or len(hashes) != DENSE_TOP_K:
        raise ValueError("dense candidate must bind three sheet hashes")
    _sha256(candidate["rank_sha256"], "dense rank hash")
    if not candidate["feasible"]:
        for index, value in enumerate(hashes):
            if value is not None:
                _sha256(value, f"dense sheet hash {index}")
        if any(
            candidate[field] is not None
            for field in ("output", "stability", "tile_majority_fraction")
        ):
            raise ValueError("infeasible dense candidate exposes a projection")
        return candidate, None
    for index, value in enumerate(hashes):
        _sha256(value, f"dense sheet hash {index}")
    if len(set(hashes)) != DENSE_TOP_K:
        raise ValueError("dense sheet hashes must be distinct")
    if candidate["output"] is None:
        raise ValueError("feasible dense candidate lacks an output")
    confidence = _confidence(candidate["stability"], "dense stability")
    majority = candidate["tile_majority_fraction"]
    if (
        type(majority) not in (int, float)
        or float(majority) not in (2 / 3, 1.0)
    ):
        raise ValueError("dense candidate lacks a strict tile majority")
    return candidate, confidence


def _retained(p0: dict[str, Any]) -> DenseDecision:
    return DenseDecision(
        "P0", "retained_p0", _snapshot_json(p0["output"]), None, None,
    )


def select_dense_candidate(
    p0: dict[str, Any], candidate: dict[str, Any], *,
    min_confidence: float, min_gain: float,
) -> DenseDecision:
    if (
        type(min_confidence) not in (int, float)
        or float(min_confidence) not in DENSE_MIN_CONFIDENCES
        or type(min_gain) not in (int, float)
        or float(min_gain) not in DENSE_MIN_GAINS
    ):
        raise ValueError("dense selector thresholds must use the frozen rule grid")
    min_confidence = float(min_confidence)
    min_gain = float(min_gain)
    p0, p0_confidence = _p0(p0)
    candidate, candidate_confidence = _candidate(candidate)
    if not candidate["feasible"] or candidate["output"] == p0["output"]:
        return _retained(p0)
    if candidate_confidence is None:
        raise AssertionError("feasible dense candidate lost its confidence")
    gain_decimal = Decimal(str(candidate_confidence)) - Decimal(str(p0_confidence))
    if (
        Decimal(str(candidate_confidence)) < Decimal(str(min_confidence))
        or gain_decimal < Decimal(str(min_gain))
    ):
        return _retained(p0)
    gain = float(gain_decimal)
    return DenseDecision(
        "DENSE", "selected_dense_evidence", _snapshot_json(candidate["output"]),
        candidate_confidence, gain,
    )
