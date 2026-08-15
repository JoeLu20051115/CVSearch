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


def _draw_kicker(report: ReportCanvas, text: str, y: float, color: colors.Color) -> float:
    c = report.canvas
    c.setFillColor(color)
    c.roundRect(ReportTheme.LEFT_MARGIN, y - 15, 72, 15, 7.5, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(ReportTheme.BODY_FONT, 8)
    c.drawCentredString(ReportTheme.LEFT_MARGIN + 36, y - 11, text)
    report.touch(y - 22)
    return y - 25


def _draw_subtitle(
    report: ReportCanvas,
    text: str,
    y: float,
    color: colors.Color = ReportTheme.NAVY,
    gap: float = 8,
) -> float:
    c = report.canvas
    c.setFillColor(color)
    c.rect(ReportTheme.LEFT_MARGIN, y - 14, 3, 14, fill=1, stroke=0)
    c.setFont(ReportTheme.BODY_FONT, 12.2)
    c.drawString(ReportTheme.LEFT_MARGIN + 9, y - 11, text)
    new_y = y - 14 - gap
    report.touch(new_y)
    return new_y


def _draw_card(
    report: ReportCanvas,
    number: str,
    title: str,
    body: str,
    x: float,
    y: float,
    width: float,
    height: float,
    color: colors.Color,
    background: colors.Color,
) -> float:
    c = report.canvas
    bottom = y - height
    c.setFillColor(background)
    c.setStrokeColor(colors.white)
    c.roundRect(x, bottom, width, height, 7, fill=1, stroke=0)
    c.setFillColor(color)
    c.circle(x + 18, y - 20, 10, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(ReportTheme.LATIN_BOLD, 8)
    c.drawCentredString(x + 18, y - 23, number)
    title_p = Paragraph(title, _style("card-title", size=10.5, leading=13, color=color))
    body_p = Paragraph(body, _style("card-body", size=8.9, leading=13, color=ReportTheme.INK))
    title_p.wrapOn(c, width - 46, height)
    title_p.drawOn(c, x + 36, y - 28)
    _, body_height = body_p.wrap(width - 24, height - 40)
    body_p.drawOn(c, x + 12, bottom + 12)
    report.touch(bottom)
    return bottom


def _draw_formula(
    report: ReportCanvas,
    formula: str,
    explanation: str,
    y: float,
    color: colors.Color,
) -> float:
    c = report.canvas
    height = 58
    bottom = y - height
    c.setFillColor(ReportTheme.PAPER)
    c.setStrokeColor(color)
    c.setLineWidth(1.0)
    c.roundRect(
        ReportTheme.LEFT_MARGIN,
        bottom,
        ReportTheme.CONTENT_WIDTH,
        height,
        6,
        fill=1,
        stroke=1,
    )
    c.setFillColor(color)
    c.setFont(ReportTheme.BODY_FONT, 13)
    c.drawString(ReportTheme.LEFT_MARGIN + 12, y - 22, formula)
    explanation_p = Paragraph(
        explanation,
        _style("formula-note", size=8.8, leading=12, color=ReportTheme.MUTED),
    )
    explanation_p.wrapOn(c, ReportTheme.CONTENT_WIDTH - 24, 24)
    explanation_p.drawOn(c, ReportTheme.LEFT_MARGIN + 12, bottom + 8)
    new_y = bottom - 10
    report.touch(new_y)
    return new_y


def draw_page_01(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    c = report.canvas
    c.setFillColor(ReportTheme.NAVY)
    c.rect(0, ReportTheme.PAGE_HEIGHT - 25 * mm, ReportTheme.PAGE_WIDTH, 25 * mm, fill=1, stroke=0)
    c.setFillColor(ReportTheme.TEAL)
    c.rect(0, 0, 7 * mm, ReportTheme.PAGE_HEIGHT, fill=1, stroke=0)
    c.setFillColor(ReportTheme.MUTED)
    c.setFont(ReportTheme.LATIN_FONT, 9)
    c.drawString(ReportTheme.LEFT_MARGIN, ReportTheme.PAGE_HEIGHT - 43 * mm, "METHOD REPORT · 2026")

    y = ReportTheme.PAGE_HEIGHT - 59 * mm
    y = draw_title(report, "从候选排序到保守验证", y, size=27, gap=3)
    y = draw_title(report, "分阶段自适应视觉搜索", y, color=ReportTheme.TEAL, size=24, gap=17)
    y = draw_paragraph(
        report,
        "一套面向高分辨率视觉问答的完整方法：先找到并排序值得看的区域，再补足尺度与上下文，在有限分支中继续观察，最后只在独立全图证据支持时替换保守答案。",
        y,
        size=12,
        leading=19,
        color=ReportTheme.NAVY,
        gap=24,
    )

    y = draw_callout(
        report,
        "一句话方法",
        "Stage 1 负责“看哪里”，Stage 2 负责“怎样看”，Stage 3 负责“是否还要继续看”，独立验证层负责“新答案能否安全替换 P0”。",
        y,
        color=ReportTheme.TEAL,
        background=ReportTheme.PALE_TEAL,
        gap=24,
    )

    box_w = (ReportTheme.CONTENT_WIDTH - 18) / 4
    labels = (
        ("1", "候选生成与排序", "Stage 1", ReportTheme.STAGE1, ReportTheme.PALE_TEAL),
        ("2", "尺度与上下文", "Stage 2", ReportTheme.STAGE2, ReportTheme.PALE_ORANGE),
        ("3", "受限分支搜索", "Stage 3", ReportTheme.STAGE3, ReportTheme.PALE_PURPLE),
        ("✓", "独立全图验证", "确认或回退", ReportTheme.VERIFY, ReportTheme.PALE_BLUE),
    )
    for index, (number, title, body, color, background) in enumerate(labels):
        x = ReportTheme.LEFT_MARGIN + index * (box_w + 6)
        _draw_card(report, number, title, body, x, y, box_w, 82, color, background)
        if index < 3:
            draw_arrow(report, x + box_w + 1, y - 41, x + box_w + 5, y - 41, ReportTheme.SLATE, 0.8)
    report.touch(y - 82)

    c.setFillColor(ReportTheme.MUTED)
    c.setFont(ReportTheme.BODY_FONT, 8.8)
    c.drawString(ReportTheme.LEFT_MARGIN, 23 * mm, "方法实现、分阶段证据与 Qwen 结果说明")


def draw_page_02(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    y = draw_header(report, "背景与贡献边界", "01 / 14")
    y = _draw_kicker(report, "WHY", y, ReportTheme.NAVY)
    y = draw_title(report, "为什么需要分阶段视觉搜索", y)
    y = draw_paragraph(
        report,
        "高分辨率图像的问题常常不是模型完全不会回答，而是关键物体太小、关系依赖周边场景，或一次裁剪没有同时保留细节和上下文。单次全图回答可能遗漏细节；无约束地不断放大，又容易丢失方位关系并增加错误替换。",
        y,
        gap=14,
    )

    y = _draw_subtitle(report, "方法目标：把三个决策分开处理", y, ReportTheme.TEAL)
    card_w = (ReportTheme.CONTENT_WIDTH - 16) / 3
    for index, (title, body) in enumerate((
        ("定位", "把搜索预算优先给与问题最相关的候选区域。"),
        ("观察", "用 ZOOM、EXPAND 与 SPLIT 补足不同类型的视觉证据。"),
        ("决策", "只有新证据足够一致且通过独立验证，才替换保守答案。"),
    )):
        _draw_card(
            report,
            str(index + 1),
            title,
            body,
            ReportTheme.LEFT_MARGIN + index * (card_w + 8),
            y,
            card_w,
            92,
            (ReportTheme.TEAL, ReportTheme.ORANGE, ReportTheme.PURPLE)[index],
            (ReportTheme.PALE_TEAL, ReportTheme.PALE_ORANGE, ReportTheme.PALE_PURPLE)[index],
        )
    y -= 108

    y = _draw_subtitle(report, "继承 CVSearch 的部分", y, ReportTheme.NAVY)
    y = draw_paragraph(
        report,
        "保留全图快速判断、问题中的目标提取、SAM 3 快速定位、失败后的语义区域树、CVSearch 初始区域排序，以及从深层到浅层的阈值搜索。它们仍是 Stage 1 的主体。",
        y,
        gap=12,
    )
    y = _draw_subtitle(report, "本文新增并评估的部分", y, ReportTheme.TEAL)
    y = draw_paragraph(
        report,
        "在同一候选池内加入 Query 感知重排；把后续 Observation 明确拆为 Stage 2 与 Stage 3；用支持度、答案一致性和不确定性控制替换；最后加入候选无关的 Qwen2.5-VL-32B 全图验证。",
        y,
        gap=12,
    )
    draw_callout(
        report,
        "贡献边界",
        "后文按阶段报告证据。Stage 1 的排序结果、Stage 2 的 Observation 修正和完整系统的端到端收益不会混为同一项提升。",
        y,
        color=ReportTheme.FALLBACK,
        background=ReportTheme.PALE_SLATE,
    )


def draw_page_03(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    y = draw_header(report, "完整系统与分工", "02 / 14")
    y = _draw_kicker(report, "PIPELINE", y, ReportTheme.NAVY)
    y = draw_title(report, "四部分各自解决一个问题", y)
    y = draw_paragraph(
        report,
        "图像与问题只沿一条主链前进。每一部分都产生明确输出；后续部分可以补充证据，但无法安全确认时必须精确返回 Stage 2 的锚点 P0。",
        y,
        gap=12,
    )

    stages = (
        ("输入", "原始图像 + 问题 + 可见选项", "材料", ReportTheme.FALLBACK, ReportTheme.PALE_SLATE),
        ("Stage 1", "全图判断 → 目标提取 → SAM 3 / 语义区域树 → 初始排序 → Query 感知重排", "输出：有序候选区域", ReportTheme.STAGE1, ReportTheme.PALE_TEAL),
        ("Stage 2", "ZOOM / EXPAND → 支持度与答案一致性 → 校准", "输出：保守锚点 P0", ReportTheme.STAGE2, ReportTheme.PALE_ORANGE),
        ("Stage 3", "SPLIT → 多分支、多尺度观察 → CONTINUE / BACKTRACK / REPLACE", "输出：安全候选或 P0", ReportTheme.STAGE3, ReportTheme.PALE_PURPLE),
        ("独立验证", "Qwen2.5-VL-32B 只看原图、问题和选项", "输出：接受候选或精确返回 P0", ReportTheme.VERIFY, ReportTheme.PALE_BLUE),
    )
    heights = (46, 70, 64, 70, 62)
    center = ReportTheme.LEFT_MARGIN + ReportTheme.CONTENT_WIDTH / 2
    for index, ((label, text, output, color, background), height) in enumerate(zip(stages, heights, strict=True)):
        label_w = 70
        c = report.canvas
        c.setFillColor(color)
        c.roundRect(ReportTheme.LEFT_MARGIN, y - height, label_w, height, 6, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(ReportTheme.BODY_FONT, 9)
        c.drawCentredString(ReportTheme.LEFT_MARGIN + label_w / 2, y - height / 2 - 3, label)
        flow_x = ReportTheme.LEFT_MARGIN + label_w + 6
        flow_w = ReportTheme.CONTENT_WIDTH - label_w - 6
        draw_flow_box(report, text + "<br/><font size='8'>" + output + "</font>", flow_x, y, flow_w, height, color, background=background, gap=0, size=9.1)
        report.touch(y - height)
        y -= height
        if index < len(stages) - 1:
            draw_arrow(report, center, y - 2, center, y - 13, ReportTheme.SLATE)
            y -= 17

    draw_callout(
        report,
        "独立验证不是 Stage 4",
        "它不再搜索区域，也不接收候选答案作为提示；它只对 Stage 3 的替换是否安全作最后约束。",
        y - 2,
        color=ReportTheme.NAVY,
        background=ReportTheme.PALE_BLUE,
    )


def draw_page_04(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    y = draw_header(report, "Stage 1 · 候选生成", "03 / 14", ReportTheme.STAGE1)
    y = _draw_kicker(report, "STAGE 1", y, ReportTheme.STAGE1)
    y = draw_title(report, "全图足够时直接回答", y, ReportTheme.STAGE1)
    y = draw_paragraph(
        report,
        "Stage 1 先判断是否真的需要搜索。系统把整张图作为根节点，针对原问题估计回答置信度；只有当置信度没有越过配置的快速阈值时，才进入目标定位。",
        y,
        gap=13,
    )

    flow_y = y
    box_w = 125
    x1 = ReportTheme.LEFT_MARGIN
    x2 = x1 + box_w + 30
    x3 = x2 + box_w + 30
    draw_flow_box(report, "整张图<br/><font size='8'>回答置信度</font>", x1, flow_y, box_w, 52, ReportTheme.NAVY, background=ReportTheme.PALE_BLUE, gap=0)
    draw_arrow(report, x1 + box_w, flow_y - 26, x2 - 4, flow_y - 26)
    draw_flow_box(report, "超过快速阈值？", x2, flow_y, box_w, 52, ReportTheme.STAGE1, background=ReportTheme.PALE_TEAL, gap=0)
    draw_arrow(report, x2 + box_w, flow_y - 26, x3 - 4, flow_y - 26)
    draw_flow_box(report, "是：直接回答<br/><font size='8'>否：继续定位</font>", x3, flow_y, box_w, 52, ReportTheme.FALLBACK, background=ReportTheme.PALE_SLATE, gap=0)
    y = flow_y - 70

    y = _draw_subtitle(report, "需要搜索时，先把问题变成可定位的目标表达", y, ReportTheme.STAGE1)
    y = draw_paragraph(
        report,
        "多模态模型从问题中抽取关键视觉目标。实现会移除带代词、指代不清的表达，并把“all dogs”一类集合表达规范为可定位的目标词；如果没有生成有效目标，则由语言处理模块从问题中提取视觉对象。",
        y,
        gap=12,
    )

    y = _draw_subtitle(report, "SAM 3 是快速定位路径", y, ReportTheme.STAGE1)
    rows = [
        ["SAM 3 结果", "Stage 1 的处理", "后续"],
        ["全部目标定位成功", "每个边界框直接变成 fast node", "跳过语义树搜索"],
        ["部分成功", "保留已找到的 fast node", "只为缺失目标搜索语义树"],
        ["未找到目标", "复用 SAM 3 backbone 特征", "构建语义区域树"],
    ]
    y = draw_table(report, rows, y, [112, 194, 147], header_color=ReportTheme.STAGE1)
    draw_callout(
        report,
        "关键点",
        "SAM 3 成功时负责“快”；失败并不终止流程，而是把同一次前向得到的视觉特征交给更细的区域搜索。",
        y,
        color=ReportTheme.STAGE1,
        background=ReportTheme.PALE_TEAL,
    )


def draw_page_05(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    y = draw_header(report, "Stage 1 · 语义区域树", "04 / 14", ReportTheme.STAGE1)
    y = _draw_kicker(report, "STAGE 1", y, ReportTheme.STAGE1)
    y = draw_title(report, "SAM 失败后构建语义区域树", y, ReportTheme.STAGE1)
    y = draw_paragraph(
        report,
        "语义区域树不是按固定网格切图。它从 SAM 3 backbone 特征出发，把语义相近且空间相邻的区域逐层合并成候选，使后续搜索既能看局部，也能沿父子关系回到更完整的上下文。",
        y,
        gap=12,
    )

    diagram_top = y
    center = ReportTheme.PAGE_WIDTH / 2
    draw_flow_box(report, "根节点：整张图", center - 78, diagram_top, 156, 38, ReportTheme.NAVY, background=ReportTheme.PALE_BLUE, gap=0)
    draw_arrow(report, center, diagram_top - 38, center, diagram_top - 53)
    level_y = diagram_top - 57
    positions = (ReportTheme.LEFT_MARGIN, center - 65, ReportTheme.PAGE_WIDTH - ReportTheme.RIGHT_MARGIN - 130)
    for index, x in enumerate(positions):
        draw_flow_box(report, f"语义子区域 {index + 1}", x, level_y, 130, 36, ReportTheme.STAGE1, background=ReportTheme.PALE_TEAL, gap=0, size=8.8)
        draw_arrow(report, center, diagram_top - 45, x + 65, level_y)
    child_y = level_y - 54
    for x in (positions[0], positions[0] + 70, positions[2] - 35, positions[2] + 35):
        draw_flow_box(report, "更细区域", x, child_y, 60, 30, ReportTheme.STAGE1, background=ReportTheme.PALE_TEAL, gap=0, size=7.7)
    report.touch(child_y - 30)
    y = child_y - 45

    y = _draw_subtitle(report, "树是怎样得到的", y, ReportTheme.STAGE1)
    rows = [
        ["步骤", "实现中的做法", "目的"],
        ["视觉原子", "SLIC 在特征图上生成约 600 个原子区域", "保留局部边界"],
        ["特征与连接", "区域平均语义特征 + 归一化位置；建立邻接图", "限制为相邻区域合并"],
        ["选择分支数", "连接约束的层次聚类，在 4–8 个子簇间选择", "适应画面复杂度"],
        ["选择标准", "轮廓系数 − 1.5 × 边界框重叠代价", "兼顾语义分离与少重叠"],
    ]
    y = draw_table(report, rows, y, [76, 245, 132], header_color=ReportTheme.STAGE1, gap=10)

    y = draw_paragraph(
        report,
        "每个节点的复杂度来自区域内语义特征与平均方向的余弦差异，并按特征幅值缩放。复杂度不足时停止继续分裂；过于简单的子区域不会保留。单目标搜索最多访问 2 层，多目标搜索最多访问 3 层。",
        y,
        size=9.4,
        leading=14,
        gap=8,
    )
    draw_callout(
        report,
        "为什么需要层级",
        "小区域利于识别文字、颜色等细节；父区域保留人物—物体、左右和前后关系。树让两种尺度共享同一条搜索路径。",
        y,
        color=ReportTheme.STAGE1,
        background=ReportTheme.PALE_TEAL,
    )


def draw_page_06(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    y = draw_header(report, "Stage 1 · CVSearch 初始排序", "05 / 14", ReportTheme.STAGE1)
    y = _draw_kicker(report, "STAGE 1", y, ReportTheme.STAGE1)
    y = draw_title(report, "CVSearch 先给出初始访问顺序", y, ReportTheme.STAGE1)
    y = draw_paragraph(
        report,
        "语义树产生的是候选集合，还需要决定先看哪一块。原始 CVSearch 用三个可解释量形成初始分数；Query 感知重排是在这一步之后执行，不会抹去这套基础顺序。",
        y,
        gap=14,
    )

    card_w = (ReportTheme.CONTENT_WIDTH - 16) / 3
    card_data = (
        ("C_current", "当前区域存在置信度", "模型判断目标是否出现在当前节点。先验概率不超过 0.4 的节点不做这次存在性评估。"),
        ("C_child", "子区域聚合信息", "取已评估子节点存在置信度的最大值，让父区域吸收其最有希望的局部证据。"),
        ("P_prior", "区域先验", "由树中相对复杂度、父节点先验和深度衰减递推得到，不使用答案标签。"),
    )
    for index, (number, title, body) in enumerate(card_data):
        _draw_card(
            report,
            str(index + 1),
            title,
            number + "：" + body,
            ReportTheme.LEFT_MARGIN + index * (card_w + 8),
            y,
            card_w,
            126,
            ReportTheme.STAGE1,
            ReportTheme.PALE_TEAL,
        )
    y -= 142

    y = _draw_formula(
        report,
        "S_cv = 0.4 C_current + 0.4 C_child + 0.2 P_prior",
        "存在置信度先从模型的 [−1, 1] 范围线性映射到 [0, 1]；同一层按 S_cv 从高到低排序。",
        y,
        ReportTheme.STAGE1,
    )

    y = _draw_subtitle(report, "最深层为什么只有两项", y, ReportTheme.STAGE1)
    y = draw_paragraph(
        report,
        "最深层没有可用的子区域信息，因此去掉 C_child，并把原来的 0.4 与 0.2 重新归一化：当前区域占 2/3，区域先验占 1/3。这样不会因为少一项而无故降低整层分数。",
        y,
        gap=12,
    )
    draw_callout(
        report,
        "初始排序的职责",
        "它回答“从树结构和目标存在性看，哪里值得先访问”。下一页的重排再回答“对当前原问题而言，哪里更相关”。",
        y,
        color=ReportTheme.FALLBACK,
        background=ReportTheme.PALE_SLATE,
    )


def draw_page_07(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    y = draw_header(report, "Stage 1 · Query 感知重排", "06 / 14", ReportTheme.STAGE1)
    y = _draw_kicker(report, "STAGE 1", y, ReportTheme.STAGE1)
    y = draw_title(report, "Query 感知重排同时考虑问题和图像", y, ReportTheme.STAGE1)
    y = draw_paragraph(
        report,
        "对 CVSearch 已保留的每个候选裁剪，重排器计算四个量，并只在当前排序事件内转成百分位。这样不同模型分数尺度不会直接相加，候选集合也不会被删减。",
        y,
        gap=10,
    )

    rows = [
        ["百分位分量", "来自哪里", "它补充的信息"],
        ["M：原问题 CLIP", "候选裁剪 × 完整原问题", "区域与真正任务是否匹配"],
        ["A：定位表达 CLIP", "至多 3 个最高增强查询分数的均值", "目标词和问题改写是否指向该区域"],
        ["C：节点复杂度", "语义区域树节点", "区域内部是否含可辨别结构"],
        ["E：边缘密度", "裁剪灰度的横纵像素差", "文字、轮廓等局部变化是否丰富"],
    ]
    y = draw_table(report, rows, y, [104, 183, 166], header_color=ReportTheme.STAGE1, gap=10)

    y = _draw_formula(
        report,
        "R = βM + (1−β)A     V = λC + (1−λ)E",
        "R 是问题相关性；V 是视觉信息量。最终排序分数为 S_query = α_eff R + (1−α_eff) V。",
        y,
        ReportTheme.STAGE1,
    )
    y = draw_paragraph(
        report,
        "最终盲测版本使用 β=0.60、α=0.65、λ=0.70。排序使用稳定降序；分数相同时保留候选原有相对次序。",
        y,
        color=ReportTheme.NAVY,
        size=9.5,
        leading=13.5,
        gap=9,
    )

    y = _draw_subtitle(report, "问题自适应权重只看问题文本", y, ReportTheme.STAGE1)
    adaptive_rows = [
        ["问题信号", "权重变化", "直观含义"],
        ["细节需求：颜色、文字、材质等", "α_eff 减 0.15", "适当增加视觉信息量的作用"],
        ["上下文需求：左右、前后、相邻等", "α_eff 加 0.15", "更强调与完整问题的语义关系"],
        ["上下文但无外观描述", "λ_eff 最多减 0.25", "边缘信息获得更多权重"],
        ["含颜色 / wearing / holding 等外观词", "解除上述 λ 折减", "保留复杂度对外观区域的作用"],
    ]
    y = draw_table(report, adaptive_rows, y, [132, 125, 196], header_color=ReportTheme.STAGE1, gap=8)
    draw_paragraph(
        report,
        "所有权重都截断到 [0, 1]；这些规则不读取正确答案、目标框或数据集身份。",
        y,
        size=8.8,
        leading=12,
        color=ReportTheme.MUTED,
        gap=0,
    )


def draw_page_08(report: ReportCanvas, evidence: Mapping[str, object]) -> None:
    del evidence
    y = draw_header(report, "Stage 1 · 按层访问与输出", "07 / 14", ReportTheme.STAGE1)
    y = _draw_kicker(report, "STAGE 1", y, ReportTheme.STAGE1)
    y = draw_title(report, "按层访问把排序转化为实际观察", y, ReportTheme.STAGE1)
    y = draw_paragraph(
        report,
        "排序完成后，系统不是同时把所有裁剪交给模型，而是按层、按候选逐个访问。每次访问都针对问题计算回答置信度，并在有限预算内寻找第一个足够可靠的区域。",
        y,
        gap=12,
    )

    steps = (
        ("1", "从最深允许层开始", "单目标最多第 2 层，多目标最多第 3 层；失败后回到更浅层。"),
        ("2", "层内先初排再重排", "CVSearch 初始分数给出基础顺序，Query 感知分数在同一候选池内重排。"),
        ("3", "逐候选计算回答置信度", "超过当前阈值即可返回；最深层可启用父区域存在性检查。"),
        ("4", "到检查点后降低阈值", "默认下降序列为 0.05、0.10、0.20，最低不越过下界，并复查已访问候选。"),
        ("5", "仍未命中则回到浅层", "第 1 层只访问排序最高的节点；完全失败时保留排序后的候选作为回退材料。"),
    )
    for number, title, body in steps:
        height = 51
        c = report.canvas
        c.setFillColor(ReportTheme.PALE_TEAL)
        c.roundRect(ReportTheme.LEFT_MARGIN, y - height, ReportTheme.CONTENT_WIDTH, height, 6, fill=1, stroke=0)
        c.setFillColor(ReportTheme.STAGE1)
        c.circle(ReportTheme.LEFT_MARGIN + 19, y - height / 2, 10, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(ReportTheme.LATIN_BOLD, 8)
        c.drawCentredString(ReportTheme.LEFT_MARGIN + 19, y - height / 2 - 3, number)
        title_p = Paragraph(title, _style("step-title", size=9.8, leading=12, color=ReportTheme.STAGE1))
        body_p = Paragraph(body, _style("step-body", size=8.5, leading=11.5, color=ReportTheme.INK))
        title_p.wrap(ReportTheme.CONTENT_WIDTH - 52, 18)
        title_p.drawOn(c, ReportTheme.LEFT_MARGIN + 38, y - 20)
        _, body_h = body_p.wrap(ReportTheme.CONTENT_WIDTH - 52, 28)
        body_p.drawOn(c, ReportTheme.LEFT_MARGIN + 38, y - height + 8)
        report.touch(y - height)
        y -= height + 7

    y = draw_callout(
        report,
        "Top-3 不是剪枝",
        "实现保留所有候选的身份集合；Top-3 用于衡量前部排序质量和限制评测预算，不表示把其余区域从搜索状态中删除。",
        y,
        color=ReportTheme.STAGE1,
        background=ReportTheme.PALE_TEAL,
        gap=9,
    )
    draw_callout(
        report,
        "Stage 1 输出",
        "有序候选区域、原图坐标、层级与树范围、初始/重排分数、已访问节点及回答置信度。Stage 2 在这些可追溯材料上选择观察尺度。",
        y,
        color=ReportTheme.FALLBACK,
        background=ReportTheme.PALE_SLATE,
        gap=0,
    )


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
