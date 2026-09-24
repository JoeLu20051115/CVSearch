# 分阶段自适应视觉搜索方法报告

该目录保存报告的可重复构建脚本、自动测试和最终 PDF。构建过程直接读取冻结实验 JSON，并对文中关键数字和归因做校验；不依赖 LaTeX。

## 构建

在仓库根目录运行：

```bash
python3 -m pip install -r reports/staged_visual_search/requirements.txt
python3 reports/staged_visual_search/build_report.py \
  --output reports/staged_visual_search/分阶段自适应视觉搜索方法报告.pdf
python3 -m unittest reports.staged_visual_search.test_report -v
```

## 证据文件

- `reproduction/evidence_gap/reports/ranking-recall-baseline.json`
- `reproduction/evidence_gap/reports/ranking-recall-v1.json`
- `reproduction/evidence_gap/reports/phase1-v6-cross-backbone.json`
- `reproduction/evidence_gap/reports/adaptive-search-v7-cross-backbone.json`
- `reproduction/evidence_gap/adaptive_search_v18/robust-transfer-candidate-free-policy.json`
- `reproduction/evidence_gap/reports/robust-transfer-mme-realworld-lite.json`

## 输出

- `分阶段自适应视觉搜索方法报告.pdf`：严格 15 页的中文方法报告。
- `build_report.py`：文字、流程图、表格、证据读取与构建入口。
- `test_report.py`：页数、文本、数值、方法术语与布局安全测试。
