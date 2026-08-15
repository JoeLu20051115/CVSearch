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
)


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


if __name__ == "__main__":
    unittest.main()
