from pathlib import Path
import tempfile
import unittest

from reports.staged_visual_search.build_report import (
    LayoutOverflow,
    ReportCanvas,
    ReportTheme,
    collect_evidence,
    draw_arrow,
    draw_callout,
    draw_flow_box,
    draw_paragraph,
    draw_table,
    draw_title,
    draw_page_01,
    draw_page_02,
    draw_page_03,
    draw_page_04,
    draw_page_05,
    draw_page_06,
    draw_page_07,
    draw_page_08,
    draw_page_09,
    draw_page_10,
    draw_page_11,
    draw_page_12,
    draw_page_13,
)

import fitz


REPO_ROOT = Path(__file__).resolve().parents[2]


class EvidenceTest(unittest.TestCase):
    def test_frozen_qwen_results(self):
        evidence = collect_evidence(REPO_ROOT)
        self.assertEqual(
            evidence["stage1_dev"],
            {
                "baseline_r3": 0.8571428571428571,
                "selected_r3": 0.9523809523809523,
                "baseline_acc": 0.8,
                "selected_acc": 0.8333333333333334,
            },
        )
        self.assertEqual(evidence["stage1_v6_qwen"]["r3_delta"], 0.0)
        self.assertEqual(evidence["stage2_qwen"]["vstar"], (17, 18))
        self.assertEqual(evidence["stage2_qwen"]["treebench"], (5, 6))
        self.assertEqual(evidence["accepted_development_qwen_delta"], 7)
        self.assertEqual(evidence["mme_qwen"]["delta"], 27)


class LayoutTest(unittest.TestCase):
    def test_layout_primitives_descend_inside_safe_area(self):
        with tempfile.TemporaryDirectory() as directory:
            report = ReportCanvas(Path(directory) / "layout.pdf")
            y0 = ReportTheme.PAGE_HEIGHT - ReportTheme.TOP_MARGIN
            y1 = draw_title(report, "中文标题 / English title", y0)
            y2 = draw_paragraph(
                report,
                "正文同时包含中文、English 与公式：S = 0.4C + 0.6P。",
                y1,
            )
            y3 = draw_callout(report, "说明", "这是不会被截断的说明框。", y2)
            y4 = draw_flow_box(
                report,
                "输入 → 判断 → 输出",
                ReportTheme.LEFT_MARGIN,
                y3,
                ReportTheme.CONTENT_WIDTH,
                34,
                ReportTheme.STAGE1,
            )
            draw_arrow(
                report,
                ReportTheme.PAGE_WIDTH / 2,
                y4 + 4,
                ReportTheme.PAGE_WIDTH / 2,
                y4 - 8,
            )
            y5 = draw_table(
                report,
                [["量", "含义", "值"], ["R@3", "前三名召回", "95.24%"]],
                y4 - 12,
                [90, 250, 90],
            )

            self.assertGreater(y0, y1)
            self.assertGreater(y1, y2)
            self.assertGreater(y2, y3)
            self.assertGreater(y3, y4)
            self.assertGreater(y4, y5)
            self.assertGreater(y5, ReportTheme.BOTTOM_MARGIN)
            report.finish_page()
            report.save()

    def test_finish_page_rejects_bottom_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            report = ReportCanvas(Path(directory) / "overflow.pdf")
            y = ReportTheme.PAGE_HEIGHT - ReportTheme.TOP_MARGIN
            for _ in range(40):
                y = draw_paragraph(
                    report,
                    "这是用于验证安全底边的长段落。" * 12,
                    y,
                    gap=4,
                )
            with self.assertRaises(LayoutOverflow):
                report.finish_page()


class Stage1PagesTest(unittest.TestCase):
    def test_first_eight_pages_follow_the_stage_one_contract(self):
        expected_headings = (
            "从候选排序到保守验证",
            "为什么需要分阶段视觉搜索",
            "四部分各自解决一个问题",
            "全图足够时直接回答",
            "SAM 失败后构建语义区域树",
            "CVSearch 先给出初始访问顺序",
            "Query 感知重排同时考虑问题和图像",
            "按层访问把排序转化为实际观察",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage1.pdf"
            report = ReportCanvas(output)
            evidence = collect_evidence(REPO_ROOT)
            for draw_page in (
                draw_page_01,
                draw_page_02,
                draw_page_03,
                draw_page_04,
                draw_page_05,
                draw_page_06,
                draw_page_07,
                draw_page_08,
            ):
                draw_page(report, evidence)
                report.finish_page()
            report.save()

            document = fitz.open(output)
            self.assertEqual(document.page_count, 8)
            pages = [page.get_text() for page in document]
            for page_text, heading in zip(pages, expected_headings, strict=True):
                self.assertIn(heading, page_text)
            text = "\n".join(pages)
            for term in (
                "0.4",
                "0.2",
                "β=0.60",
                "α=0.65",
                "λ=0.70",
                "Top-3 不是剪枝",
            ):
                self.assertIn(term, text)


class LaterPagesTest(unittest.TestCase):
    def test_later_stage_pages_keep_uncertainty_and_fallback_controls(self):
        expected_headings = (
            "ZOOM 补细节，EXPAND 补上下文",
            "Stage 2 把可回退答案保存为 P0",
            "SPLIT 只在局部位置仍模糊时继续细分",
            "不确定性控制决定继续、回退或提出替换",
            "独立全图答案负责最后的否决与确认",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "later.pdf"
            report = ReportCanvas(output)
            evidence = collect_evidence(REPO_ROOT)
            for draw_page in (
                draw_page_09,
                draw_page_10,
                draw_page_11,
                draw_page_12,
                draw_page_13,
            ):
                draw_page(report, evidence)
                report.finish_page()
            report.save()

            document = fitz.open(output)
            self.assertEqual(document.page_count, 5)
            pages = [page.get_text() for page in document]
            for page_text, heading in zip(pages, expected_headings, strict=True):
                self.assertIn(heading, page_text)
            text = "\n".join(pages)
            for term in (
                "1-P(sufficient)",
                "CONTINUE",
                "BACKTRACK",
                "REPLACE",
                "STOP_P0",
                "FALLBACK_P0",
                "Qwen2.5-VL-32B",
                "0.6",
                "0.4",
            ):
                self.assertIn(term, text)


if __name__ == "__main__":
    unittest.main()
