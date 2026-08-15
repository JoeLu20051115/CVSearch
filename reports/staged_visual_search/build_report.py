#!/usr/bin/env python3
"""Build the staged adaptive visual-search method report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


EVIDENCE_PATHS = {
    "stage1_baseline": "reproduction/evidence_gap/reports/ranking-recall-baseline.json",
    "stage1_v1": "reproduction/evidence_gap/reports/ranking-recall-v1.json",
    "stage1_v6": "reproduction/evidence_gap/reports/phase1-v6-cross-backbone.json",
    "stage2": "reproduction/evidence_gap/reports/adaptive-search-v7-cross-backbone.json",
    "accepted_policy": (
        "reproduction/evidence_gap/adaptive_search_v18/"
        "robust-transfer-candidate-free-policy.json"
    ),
    "mme": "reproduction/evidence_gap/reports/robust-transfer-mme-realworld-lite.json",
}


pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


class LayoutOverflow(RuntimeError):
    """Raised when content crosses the report's safe bottom boundary."""


class ReportTheme:
    PAGE_WIDTH, PAGE_HEIGHT = A4
    LEFT_MARGIN = 25 * mm
    RIGHT_MARGIN = 25 * mm
    TOP_MARGIN = 22 * mm
    BOTTOM_MARGIN = 18 * mm
    CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

    NAVY = colors.HexColor("#173A5E")
    TEAL = colors.HexColor("#087F7B")
    ORANGE = colors.HexColor("#D97A16")
    PURPLE = colors.HexColor("#6B4C9A")
    SLATE = colors.HexColor("#7890A8")
    INK = colors.HexColor("#233140")
    MUTED = colors.HexColor("#617184")
    LINE = colors.HexColor("#CFD8E2")
    PAPER = colors.HexColor("#FBFCFE")
    PALE_BLUE = colors.HexColor("#EDF4F8")
    PALE_TEAL = colors.HexColor("#EAF6F4")
    PALE_ORANGE = colors.HexColor("#FCF3E8")
    PALE_PURPLE = colors.HexColor("#F2EEF7")
    PALE_SLATE = colors.HexColor("#F0F3F6")

    STAGE1 = TEAL
    STAGE2 = ORANGE
    STAGE3 = PURPLE
    VERIFY = NAVY
    FALLBACK = SLATE

    BODY_FONT = "STSong-Light"
    LATIN_FONT = "Helvetica"
    LATIN_BOLD = "Helvetica-Bold"
    BODY_SIZE = 10.2
    BODY_LEADING = 15.2


class ReportCanvas:
    """Small page-aware wrapper around ReportLab's canvas."""

    def __init__(self, output_path: Path, title: str = "") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.canvas = Canvas(str(self.output_path), pagesize=A4)
        self.canvas.setTitle(title)
        self.page_index = 1
        self.lowest_y = ReportTheme.PAGE_HEIGHT - ReportTheme.TOP_MARGIN

    def touch(self, y: float) -> None:
        self.lowest_y = min(self.lowest_y, y)

    def finish_page(self, numbered: bool | None = None) -> None:
        if self.lowest_y < ReportTheme.BOTTOM_MARGIN:
            raise LayoutOverflow(
                f"Page {self.page_index} crossed the safe bottom: "
                f"{self.lowest_y:.1f} < {ReportTheme.BOTTOM_MARGIN:.1f}"
            )
        if numbered is None:
            numbered = self.page_index > 1
        if numbered:
            self.canvas.setFillColor(ReportTheme.MUTED)
            self.canvas.setFont(ReportTheme.LATIN_FONT, 8.5)
            self.canvas.drawCentredString(
                ReportTheme.PAGE_WIDTH / 2,
                9.5 * mm,
                str(self.page_index - 1),
            )
        self.canvas.showPage()
        self.page_index += 1
        self.lowest_y = ReportTheme.PAGE_HEIGHT - ReportTheme.TOP_MARGIN

    def save(self) -> None:
        self.canvas.save()


def _style(
    name: str,
    *,
    size: float = ReportTheme.BODY_SIZE,
    leading: float = ReportTheme.BODY_LEADING,
    color: colors.Color = ReportTheme.INK,
    alignment: int = TA_LEFT,
    font: str = ReportTheme.BODY_FONT,
) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName=font,
        fontSize=size,
        leading=leading,
        textColor=color,
        alignment=alignment,
        spaceAfter=0,
        spaceBefore=0,
        splitLongWords=False,
        allowWidows=0,
        allowOrphans=0,
    )


def draw_header(
    report: ReportCanvas,
    section: str,
    page_label: str = "",
    color: colors.Color = ReportTheme.NAVY,
) -> float:
    c = report.canvas
    y = ReportTheme.PAGE_HEIGHT - 14 * mm
    c.setFillColor(color)
    c.setFont(ReportTheme.BODY_FONT, 8.5)
    c.drawString(ReportTheme.LEFT_MARGIN, y, section)
    if page_label:
        c.setFillColor(ReportTheme.MUTED)
        c.drawRightString(
            ReportTheme.PAGE_WIDTH - ReportTheme.RIGHT_MARGIN, y, page_label
        )
    c.setStrokeColor(ReportTheme.LINE)
    c.setLineWidth(0.7)
    c.line(
        ReportTheme.LEFT_MARGIN,
        y - 3.5 * mm,
        ReportTheme.PAGE_WIDTH - ReportTheme.RIGHT_MARGIN,
        y - 3.5 * mm,
    )
    return ReportTheme.PAGE_HEIGHT - ReportTheme.TOP_MARGIN


def draw_title(
    report: ReportCanvas,
    text: str,
    y: float,
    color: colors.Color = ReportTheme.NAVY,
    size: float = 22,
    gap: float = 13,
) -> float:
    paragraph = Paragraph(text, _style("title", size=size, leading=size * 1.25, color=color))
    width = ReportTheme.CONTENT_WIDTH
    _, height = paragraph.wrap(width, ReportTheme.PAGE_HEIGHT)
    paragraph.drawOn(report.canvas, ReportTheme.LEFT_MARGIN, y - height)
    new_y = y - height - gap
    report.touch(new_y)
    return new_y


def draw_paragraph(
    report: ReportCanvas,
    text: str,
    y: float,
    *,
    x: float = ReportTheme.LEFT_MARGIN,
    width: float = ReportTheme.CONTENT_WIDTH,
    size: float = ReportTheme.BODY_SIZE,
    leading: float = ReportTheme.BODY_LEADING,
    color: colors.Color = ReportTheme.INK,
    alignment: int = TA_LEFT,
    gap: float = 10,
) -> float:
    paragraph = Paragraph(
        text,
        _style(
            "body",
            size=size,
            leading=leading,
            color=color,
            alignment=alignment,
        ),
    )
    _, height = paragraph.wrap(width, ReportTheme.PAGE_HEIGHT)
    paragraph.drawOn(report.canvas, x, y - height)
    new_y = y - height - gap
    report.touch(new_y)
    return new_y


def draw_callout(
    report: ReportCanvas,
    title: str,
    body: str,
    y: float,
    *,
    color: colors.Color = ReportTheme.TEAL,
    background: colors.Color = ReportTheme.PALE_TEAL,
    x: float = ReportTheme.LEFT_MARGIN,
    width: float = ReportTheme.CONTENT_WIDTH,
    gap: float = 12,
) -> float:
    padding = 10
    title_p = Paragraph(title, _style("callout-title", size=11.3, leading=14, color=color))
    body_p = Paragraph(body, _style("callout-body", size=9.5, leading=14, color=ReportTheme.INK))
    inner_width = width - 2 * padding
    _, title_height = title_p.wrap(inner_width, ReportTheme.PAGE_HEIGHT)
    _, body_height = body_p.wrap(inner_width, ReportTheme.PAGE_HEIGHT)
    height = padding + title_height + 4 + body_height + padding
    bottom = y - height
    c = report.canvas
    c.setFillColor(background)
    c.setStrokeColor(color)
    c.setLineWidth(0.8)
    c.roundRect(x, bottom, width, height, 6, fill=1, stroke=1)
    title_p.drawOn(c, x + padding, y - padding - title_height)
    body_p.drawOn(c, x + padding, bottom + padding)
    new_y = bottom - gap
    report.touch(new_y)
    return new_y


def draw_flow_box(
    report: ReportCanvas,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    color: colors.Color,
    *,
    background: colors.Color | None = None,
    gap: float = 8,
    size: float = 9.5,
) -> float:
    background = background or colors.Color(
        0.93 + color.red * 0.07,
        0.93 + color.green * 0.07,
        0.93 + color.blue * 0.07,
    )
    bottom = y - height
    c = report.canvas
    c.setFillColor(background)
    c.setStrokeColor(color)
    c.setLineWidth(1.0)
    c.roundRect(x, bottom, width, height, 7, fill=1, stroke=1)
    paragraph = Paragraph(
        text,
        _style("flow", size=size, leading=size * 1.28, color=color, alignment=TA_CENTER),
    )
    _, text_height = paragraph.wrap(width - 12, height - 6)
    paragraph.drawOn(c, x + 6, bottom + (height - text_height) / 2)
    new_y = bottom - gap
    report.touch(new_y)
    return new_y


def draw_arrow(
    report: ReportCanvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: colors.Color = ReportTheme.SLATE,
    width: float = 1.2,
) -> None:
    c = report.canvas
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = 5
    if abs(y2 - y1) >= abs(x2 - x1):
        direction = 1 if y2 > y1 else -1
        c.line(x2, y2, x2 - angle, y2 - direction * angle)
        c.line(x2, y2, x2 + angle, y2 - direction * angle)
    else:
        direction = 1 if x2 > x1 else -1
        c.line(x2, y2, x2 - direction * angle, y2 - angle)
        c.line(x2, y2, x2 - direction * angle, y2 + angle)
    report.touch(min(y1, y2) - angle)


def draw_table(
    report: ReportCanvas,
    rows: list[list[str]],
    y: float,
    column_widths: list[float],
    *,
    x: float = ReportTheme.LEFT_MARGIN,
    gap: float = 12,
    header_color: colors.Color = ReportTheme.NAVY,
) -> float:
    if not rows:
        return y
    header_style = _style(
        "table-header", size=8.8, leading=11.5, color=header_color
    )
    body_style = _style("table-body", size=8.2, leading=11.2, color=ReportTheme.INK)
    cells = [
        [Paragraph(str(cell), header_style if row_index == 0 else body_style) for cell in row]
        for row_index, row in enumerate(rows)
    ]
    table = Table(cells, colWidths=column_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEABOVE", (0, 0), (-1, 0), 1.0, header_color),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, header_color),
                ("LINEBELOW", (0, -1), (-1, -1), 1.0, header_color),
                ("BACKGROUND", (0, 0), (-1, 0), ReportTheme.PALE_BLUE),
            ]
        )
    )
    width, height = table.wrap(sum(column_widths), ReportTheme.PAGE_HEIGHT)
    bottom = y - height
    table.drawOn(report.canvas, x, bottom)
    new_y = bottom - gap
    report.touch(new_y)
    return new_y


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Evidence must be a JSON object: {path}")
    return value


def _field(value: Mapping[str, Any], path: str, expected: type) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"Missing evidence field: {path}")
        current = current[part]
    if expected is float:
        if type(current) not in (int, float):
            raise ValueError(f"Evidence field is not numeric: {path}")
        return float(current)
    if type(current) is not expected:
        raise ValueError(f"Evidence field has wrong type: {path}")
    return current


def collect_evidence(repo_root: Path) -> dict[str, object]:
    """Read the frozen experiment reports used by pages 14 and 15."""

    reports = {
        name: _load_object(repo_root / relative_path)
        for name, relative_path in EVIDENCE_PATHS.items()
    }

    baseline = reports["stage1_baseline"]
    selected = reports["stage1_v1"]
    v6 = reports["stage1_v6"]
    stage2 = reports["stage2"]
    accepted = reports["accepted_policy"]
    mme = reports["mme"]

    qwen_v6 = _field(v6, "results.qwen2_5_vl_7b", dict)
    qwen_stage2 = _field(stage2, "cells.qwen2_5_vl_7b", dict)
    accepted_metrics = _field(accepted, "accepted_outer_checkpoint.metrics", dict)
    mme_qwen = _field(mme, "backbones.qwen", dict)
    mme_internvl = _field(mme, "backbones.internvl", dict)

    evidence: dict[str, object] = {
        "stage1_dev": {
            "baseline_r3": _field(baseline, "topic.recall_at.3", float),
            "selected_r3": _field(selected, "topic.recall_at.3", float),
            "baseline_acc": _field(baseline, "answer_accuracy", float),
            "selected_acc": _field(selected, "answer_accuracy", float),
        },
        "stage1_dev_r1": (
            _field(baseline, "topic.recall_at.1", float),
            _field(selected, "topic.recall_at.1", float),
        ),
        "stage1_v6_qwen": {
            "baseline_r1": _field(qwen_v6, "original.recall_at_1", float),
            "selected_r1": _field(qwen_v6, "v6.recall_at_1", float),
            "baseline_r3": _field(qwen_v6, "original.recall_at_3", float),
            "selected_r3": _field(qwen_v6, "v6.recall_at_3", float),
            "r3_delta": _field(qwen_v6, "recall_at_3_delta", float),
            "baseline_acc": _field(qwen_v6, "original.answer_accuracy", float),
            "selected_acc": _field(qwen_v6, "v6.answer_accuracy", float),
            "acc_delta": _field(qwen_v6, "answer_accuracy_delta", float),
        },
        "stage2_qwen": {
            "vstar": (
                _field(qwen_stage2, "vstar.baseline_correct", int),
                _field(qwen_stage2, "vstar.selected_correct", int),
            ),
            "treebench": (
                _field(qwen_stage2, "treebench.baseline_correct", int),
                _field(qwen_stage2, "treebench.selected_correct", int),
            ),
            "hr_bench_4k": (
                _field(qwen_stage2, "hr_bench_4k.baseline_correct", int),
                _field(qwen_stage2, "hr_bench_4k.selected_correct", int),
            ),
            "hr_bench_8k": (
                _field(qwen_stage2, "hr_bench_8k.baseline_correct", int),
                _field(qwen_stage2, "hr_bench_8k.selected_correct", int),
            ),
        },
        "stage2_aggregate": {
            "corrections": _field(stage2, "aggregate.corrections", int),
            "corruptions": _field(stage2, "aggregate.corruptions", int),
        },
        "accepted_development_qwen_delta": _field(
            accepted_metrics, "backbone_deltas.qwen", int
        ),
        "accepted_development": {
            "net_gain": _field(accepted_metrics, "net_gain", int),
            "corrections": _field(accepted_metrics, "corrections", int),
            "corruptions": _field(accepted_metrics, "corruptions", int),
            "qwen_delta": _field(accepted_metrics, "backbone_deltas.qwen", int),
            "internvl_delta": _field(
                accepted_metrics, "backbone_deltas.internvl", int
            ),
            "topics": _field(accepted_metrics, "topics", int),
            "official_units": _field(accepted_metrics, "official_units", int),
            "mean_observations": _field(accepted_metrics, "mean_observations", float),
        },
        "mme_qwen": {
            key: _field(mme_qwen, key, float if key == "mean_observations" else int)
            for key in (
                "topics",
                "baseline_correct",
                "robust_correct",
                "delta",
                "corrections",
                "corruptions",
                "selections",
                "runtime_failures",
                "observations",
                "mean_observations",
            )
        },
        "mme_internvl": {
            key: _field(
                mme_internvl, key, float if key == "mean_observations" else int
            )
            for key in (
                "topics",
                "baseline_correct",
                "robust_correct",
                "delta",
                "corrections",
                "corruptions",
                "runtime_failures",
                "mean_observations",
            )
        },
        "mme_evaluation_count": _field(mme, "evaluation_count", int),
        "mme_gate_passed": _field(mme, "gate.passed", bool),
    }
    validate_claims(evidence)
    return evidence


def validate_claims(evidence: Mapping[str, object]) -> None:
    """Reject evidence drift that would make the report's claims inaccurate."""

    stage1_v6 = _field(evidence, "stage1_v6_qwen", dict)
    if _field(stage1_v6, "r3_delta", float) < 0:
        raise ValueError("Stage 1 v6 Qwen Recall@3 must not regress")
    if _field(stage1_v6, "acc_delta", float) < 0:
        raise ValueError("Stage 1 v6 Qwen accuracy must not regress")

    stage2 = _field(evidence, "stage2_qwen", dict)
    for unit_name, pair in stage2.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Malformed Stage 2 score pair: {unit_name}")
        if type(pair[0]) is not int or type(pair[1]) is not int:
            raise ValueError(f"Stage 2 score pair must contain integers: {unit_name}")
        if pair[1] < pair[0]:
            raise ValueError(f"Stage 2 Qwen result regressed: {unit_name}")

    accepted = _field(evidence, "accepted_development", dict)
    if _field(accepted, "qwen_delta", int) != 7:
        raise ValueError("Accepted development policy Qwen delta drifted from +7")
    if _field(accepted, "corruptions", int) != 0:
        raise ValueError("Accepted development policy must have zero corruptions")

    if _field(evidence, "mme_evaluation_count", int) != 1:
        raise ValueError("MME-RealWorld-Lite must remain a single sealed evaluation")
    if not _field(evidence, "mme_gate_passed", bool):
        raise ValueError("MME-RealWorld-Lite result did not pass its frozen gate")
    mme_qwen = _field(evidence, "mme_qwen", dict)
    if _field(mme_qwen, "delta", int) != 27:
        raise ValueError("MME Qwen end-to-end delta drifted from +27")
    if _field(mme_qwen, "runtime_failures", int) != 0:
        raise ValueError("MME Qwen evaluation contains runtime failures")
