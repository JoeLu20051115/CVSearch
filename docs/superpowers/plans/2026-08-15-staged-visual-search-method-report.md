# 分阶段自适应视觉搜索方法报告 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份约 15 页、可重复构建的中文方法报告，完整解释 Stage 1、Stage 2、Stage 3、不确定性控制和独立验证层，并准确呈现 Qwen 的分阶段与端到端结果。

**Architecture:** 使用一个 ReportLab 源脚本集中管理页面主题、文字、流程图和表格；构建时直接读取冻结 JSON 证据并对关键数字做断言，避免手工抄写产生漂移。单元测试分别验证证据提取、阶段术语、页数、文本可检索性和布局溢出，最终使用 PyMuPDF 渲染逐页检查。

**Tech Stack:** Python 3.11、ReportLab、PyMuPDF、标准库 `unittest`、现有 JSON/JSONL 实验材料。

## Global Constraints

- 输出目录固定为 `reports/staged_visual_search/`。
- PDF 标题固定为“从候选排序到保守验证：分阶段自适应视觉搜索”。
- 正文使用中文；首次出现的接口名保留英文并立刻解释。
- 页面顺序遵循已批准的 15 页设计，最终 PDF 严格为 15 页。
- Stage 1、Stage 2、Stage 3 和独立验证层使用稳定且不同的颜色；`P0` 与回退使用灰蓝色。
- 只使用已存在于代码、配置、设计稿或冻结报告中的术语和公式。
- Stage 1 开发结果、v6 盲测、Stage 2 结果、最终开发策略和 MME 结果必须从指定 JSON 文件读取并逐项验证。
- MME 的 `+27` 只表述为完整系统端到端收益，不归因于单一阶段。
- 原 PDF 的 DiNCo-inspired 三次问询公式不作为当前实现写入。
- 不修改任何已有实验文件，不安装依赖到仓库源码目录，不提交临时渲染图片。

---

### Task 1: 建立证据读取与主张约束

**Files:**
- Create: `reports/staged_visual_search/build_report.py`
- Create: `reports/staged_visual_search/test_report.py`

**Interfaces:**
- Consumes: 仓库根目录 `Path` 和六个冻结 JSON 材料。
- Produces: `collect_evidence(repo_root: Path) -> dict[str, object]`，返回页面 14–15 所需的所有数字；`validate_claims(evidence: Mapping[str, object]) -> None` 在数字或归因漂移时抛出 `ValueError`。

- [ ] **Step 1: 写证据提取失败测试**

测试必须断言：

```python
class EvidenceTest(unittest.TestCase):
    def test_frozen_qwen_results(self):
        evidence = collect_evidence(REPO_ROOT)
        self.assertEqual(evidence["stage1_dev"], {
            "baseline_r3": 0.8571428571428571,
            "selected_r3": 0.9523809523809523,
            "baseline_acc": 0.8,
            "selected_acc": 0.8333333333333334,
        })
        self.assertEqual(evidence["stage1_v6_qwen"]["r3_delta"], 0.0)
        self.assertEqual(evidence["stage2_qwen"]["vstar"], (17, 18))
        self.assertEqual(evidence["stage2_qwen"]["treebench"], (5, 6))
        self.assertEqual(evidence["accepted_development_qwen_delta"], 7)
        self.assertEqual(evidence["mme_qwen"]["delta"], 27)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest reports.staged_visual_search.test_report.EvidenceTest -v
```

Expected: 因 `build_report.py` 尚不存在而导入失败。

- [ ] **Step 3: 实现严格证据提取**

`collect_evidence` 必须读取：

```python
EVIDENCE_PATHS = {
    "stage1_baseline": "reproduction/evidence_gap/reports/ranking-recall-baseline.json",
    "stage1_v1": "reproduction/evidence_gap/reports/ranking-recall-v1.json",
    "stage1_v6": "reproduction/evidence_gap/reports/phase1-v6-cross-backbone.json",
    "stage2": "reproduction/evidence_gap/reports/adaptive-search-v7-cross-backbone.json",
    "accepted_policy": "reproduction/evidence_gap/adaptive_search_v18/robust-transfer-candidate-free-policy.json",
    "mme": "reproduction/evidence_gap/reports/robust-transfer-mme-realworld-lite.json",
}
```

它必须验证 JSON 顶层类型、精确字段类型和以下不变量：Stage 1 v6 Qwen `Recall@3` 与准确率均不下降；Stage 2 Qwen 的四个数据单元均不下降；接受策略的 Qwen delta 为 7 且总破坏数为 0；MME 只评测一次、Qwen delta 为 27、运行失败数为 0。

- [ ] **Step 4: 运行证据测试并确认通过**

Run 同 Step 2。Expected: `OK`。

- [ ] **Step 5: 提交证据合同**

```bash
git add reports/staged_visual_search/build_report.py reports/staged_visual_search/test_report.py
git commit -m "test: lock method report evidence"
```

---

### Task 2: 实现 A4 主题、文字布局和流程图原语

**Files:**
- Modify: `reports/staged_visual_search/build_report.py`
- Modify: `reports/staged_visual_search/test_report.py`
- Create: `reports/staged_visual_search/requirements.txt`

**Interfaces:**
- Produces: `ReportTheme`、`ReportCanvas`、`draw_header`、`draw_title`、`draw_paragraph`、`draw_callout`、`draw_flow_box`、`draw_arrow`、`draw_table`。
- `ReportCanvas.finish_page()` 在当前页内容低于安全底边时抛出 `LayoutOverflow`。

- [ ] **Step 1: 写布局原语测试**

测试一页中文、英文、公式、说明框、流程框和三列表格，断言所有绘制函数返回严格下降且高于底边的 y 坐标；向一页重复写入长段落时必须抛出 `LayoutOverflow`。

- [ ] **Step 2: 安装临时构建依赖并确认测试失败**

Run:

```bash
python3 -m pip install --target /tmp/codex-staged-report-tools reportlab==4.4.3 pymupdf==1.26.3
PYTHONPATH=/tmp/codex-staged-report-tools /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest reports.staged_visual_search.test_report.LayoutTest -v
```

Expected: 布局类和函数尚不存在。

- [ ] **Step 3: 实现最小布局系统**

- 使用 A4 页面，左右页边距 25 mm，正文安全底边 18 mm。
- 注册 `STSong-Light` 中文 CID 字体、`Helvetica` 和 `Helvetica-Bold`。
- 颜色固定为：深蓝 `#173A5E`、青绿 `#087F7B`、橙色 `#D97A16`、紫色 `#6B4C9A`、灰蓝 `#7890A8`。
- 正文字号 10.2 pt、行距 15.2 pt；页标题 22 pt；页眉 8.5 pt；表格最低字号 8.2 pt。
- `draw_paragraph` 使用 ReportLab `Paragraph` 测量和绘制，不允许截断文本。
- `draw_table` 使用 `Table` 和 `TableStyle`，只画顶线、表头底线和底线，不画纵线。

- [ ] **Step 4: 运行布局测试并确认通过**

Run 同 Step 2 的 unittest 命令。Expected: `OK`。

- [ ] **Step 5: 写入可重复依赖**

`requirements.txt` 精确包含：

```text
reportlab==4.4.3
pymupdf==1.26.3
```

- [ ] **Step 6: 提交布局基础**

```bash
git add reports/staged_visual_search/build_report.py reports/staged_visual_search/test_report.py reports/staged_visual_search/requirements.txt
git commit -m "feat: add staged report layout system"
```

---

### Task 3: 写作并绘制第 1–8 页 Stage 1

**Files:**
- Modify: `reports/staged_visual_search/build_report.py`
- Modify: `reports/staged_visual_search/test_report.py`

**Interfaces:**
- Produces: `draw_page_01` 至 `draw_page_08`；每个函数接收 `(canvas: ReportCanvas, evidence: Mapping[str, object])`。

- [ ] **Step 1: 写页面内容合同测试**

构建前八页到临时 PDF，提取文本并断言按顺序包含：

```python
EXPECTED_STAGE1_HEADINGS = (
    "从候选排序到保守验证",
    "为什么需要分阶段视觉搜索",
    "四部分各自解决一个问题",
    "全图足够时直接回答",
    "SAM 失败后构建语义区域树",
    "CVSearch 先给出初始访问顺序",
    "Query 感知重排同时考虑问题和图像",
    "按层访问把排序转化为实际观察",
)
```

并断言正文包含 `0.4`、`0.4`、`0.2`、`β=0.60`、`α=0.65`、`λ=0.70`、`Top-3 不是剪枝`。

- [ ] **Step 2: 运行页面测试并确认失败**

Run:

```bash
PYTHONPATH=/tmp/codex-staged-report-tools /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest reports.staged_visual_search.test_report.Stage1PagesTest -v
```

Expected: 页面函数尚不存在。

- [ ] **Step 3: 完成第 1–3 页**

- 第 1 页使用原 PDF 的封面层级、一句话方法框和四段缩略流程。
- 第 2 页用“问题—继承—改动”三段说明贡献边界。
- 第 3 页绘制完整四阶段纵向流程，并在每个阶段右侧写输入和输出。

- [ ] **Step 4: 完成第 4–6 页**

- 第 4 页按全图置信度、目标提取、SAM 成功/失败路径写作。
- 第 5 页绘制语义区域树并解释 600 个视觉原子、4–8 个子簇、复杂度停止和 2/3 层深度。
- 第 6 页先解释三个初始排序量，再展示 `S_cv = 0.4 C_current + 0.4 C_child + 0.2 P_prior`，说明最深层的两项重归一化。

- [ ] **Step 5: 完成第 7–8 页**

- 第 7 页解释四个百分位分量、三步融合公式和 v6 问题自适应权重。
- 第 8 页解释从深到浅、逐候选访问、阈值下降、父区域检查和 Stage 1 输出材料。

- [ ] **Step 6: 运行页面合同和布局测试**

Run:

```bash
PYTHONPATH=/tmp/codex-staged-report-tools /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest reports.staged_visual_search.test_report.Stage1PagesTest reports.staged_visual_search.test_report.LayoutTest -v
```

Expected: 全部 `OK`，无 `LayoutOverflow`。

- [ ] **Step 7: 提交 Stage 1 页面**

```bash
git add reports/staged_visual_search/build_report.py reports/staged_visual_search/test_report.py
git commit -m "docs: write stage one report pages"
```

---

### Task 4: 写作并绘制第 9–13 页后续阶段

**Files:**
- Modify: `reports/staged_visual_search/build_report.py`
- Modify: `reports/staged_visual_search/test_report.py`

**Interfaces:**
- Produces: `draw_page_09` 至 `draw_page_13`。

- [ ] **Step 1: 写后续阶段内容合同测试**

构建第 9–13 页并断言顺序包含：

```python
EXPECTED_LATER_HEADINGS = (
    "ZOOM 补细节，EXPAND 补上下文",
    "Stage 2 把可回退答案保存为 P0",
    "SPLIT 只在局部位置仍模糊时继续细分",
    "不确定性控制决定继续、回退或提出替换",
    "独立全图答案负责最后的否决与确认",
)
```

文本还必须包含 `1-P(sufficient)`、`CONTINUE`、`BACKTRACK`、`REPLACE`、`STOP_P0`、`FALLBACK_P0`、`Qwen2.5-VL-32B`、`0.6` 和 `0.4`。

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
PYTHONPATH=/tmp/codex-staged-report-tools /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest reports.staged_visual_search.test_report.LaterPagesTest -v
```

Expected: 页面函数尚不存在。

- [ ] **Step 3: 完成 Stage 2 两页**

- 第 9 页用同一示例区域并排画 `ZOOM` 与 `EXPAND`，明确两者不自动修改答案。
- 第 10 页写支持概率、原始支持度、校准、答案一致性、支持不确定性和 `P0` 精确重建。

- [ ] **Step 4: 完成 Stage 3 两页**

- 第 11 页画有限分支与 tight/medium/context 多尺度观察，解释双渲染支持要求和固定预算。
- 第 12 页画 `OBSERVE → CONTINUE → BACKTRACK/REPLACE/STOP_P0` 状态图，解释答案不确定性、agreement、minimum support、support gain 和 P0 conflict margin 的职责。

- [ ] **Step 5: 完成独立验证页**

- 第 13 页画候选无关的 Qwen2.5-VL-32B 全图答题分支。
- 明确独立答案不接收 `P0` 或候选；高置信冲突否决已接受候选，严格一致确认回退后的 observation-8 提议，其余路径返回 `P0`。

- [ ] **Step 6: 运行后续页面和布局测试**

Expected: 全部 `OK`，无术语缺失和布局溢出。

- [ ] **Step 7: 提交后续阶段页面**

```bash
git add reports/staged_visual_search/build_report.py reports/staged_visual_search/test_report.py
git commit -m "docs: explain observation and robust selection stages"
```

---

### Task 5: 写作结果页、构建入口和重建说明

**Files:**
- Modify: `reports/staged_visual_search/build_report.py`
- Modify: `reports/staged_visual_search/test_report.py`
- Create: `reports/staged_visual_search/README.md`
- Create: `reports/staged_visual_search/分阶段自适应视觉搜索方法报告.pdf`

**Interfaces:**
- Produces: `draw_page_14`、`draw_page_15`、`build_report(repo_root: Path, output_path: Path) -> None` 和 CLI `python build_report.py --output <path>`。

- [ ] **Step 1: 写完整报告合同测试**

测试构建完整 PDF 并断言：

- 页数严格为 15；
- PDF metadata 标题正确；
- 每页都能提取非空文本；
- 页码 1–14 只出现在正文页，封面无页码；
- 文本包含 `85.71%`、`95.24%`、`80.00%`、`83.33%`、`17/20`、`18/20`、`5/12`、`6/12`、`880/1919`、`907/1919`、`+27`、`49 次修正` 和 `22 次破坏`；
- MME 段落附近包含“完整系统”，不包含“Stage 1 带来 27”。

- [ ] **Step 2: 运行完整合同并确认失败**

Expected: 结果页和构建入口尚不存在。

- [ ] **Step 3: 完成第 14 页分阶段结果**

用一个四行表分别报告 Stage 1 开发、Stage 1 v6 盲测、Stage 2 声明测试和最终开发策略；表下明确不同分区的数字不能相加。披露 Recall@1 的下降和已打开开发数据的限制。

- [ ] **Step 4: 完成第 15 页 MME 与结论**

用 Qwen 主结果卡展示 `880/1919 → 907/1919`、`+27`、`+1.41` 个百分点、49/22 修正破坏和 8.41 平均观察；用一行补充 InternVL `+18`。结尾列出一次封存评测、32B 额外计算和不能单阶段归因三项限制。

- [ ] **Step 5: 实现 CLI 和 README**

README 必须给出：

```bash
python3 -m pip install -r reports/staged_visual_search/requirements.txt
python3 reports/staged_visual_search/build_report.py \
  --output reports/staged_visual_search/分阶段自适应视觉搜索方法报告.pdf
python3 -m unittest reports.staged_visual_search.test_report -v
```

并列出证据文件和输出文件，不要求 LaTeX。

- [ ] **Step 6: 构建 PDF 并运行完整测试**

Run:

```bash
PYTHONPATH=/tmp/codex-staged-report-tools /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 reports/staged_visual_search/build_report.py --output reports/staged_visual_search/分阶段自适应视觉搜索方法报告.pdf
PYTHONPATH=/tmp/codex-staged-report-tools /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest reports.staged_visual_search.test_report -v
```

Expected: PDF 生成，全部测试 `OK`。

- [ ] **Step 7: 提交完整报告**

```bash
git add reports/staged_visual_search
git commit -m "docs: add staged visual search method report"
```

---

### Task 6: 视觉核验与学术主张自审

**Files:**
- Modify if needed: `reports/staged_visual_search/build_report.py`
- Modify if needed: `reports/staged_visual_search/分阶段自适应视觉搜索方法报告.pdf`
- Modify if needed: `reports/staged_visual_search/test_report.py`

**Interfaces:**
- Produces: 最终视觉无缺陷 PDF 和通过的五维自审记录（保留在终端核验输出，不增加正文页）。

- [ ] **Step 1: 渲染全部页面**

使用 PyMuPDF 将 15 页以 150 dpi 渲染到 `/tmp/staged-visual-search-report-pages/`，生成一张低分辨率联系表，并单独保留第 1、3、7、10、12、13、14、15 页高分辨率图。

- [ ] **Step 2: 检查视觉缺陷**

逐页检查：中文缺字、文本重叠、流程箭头穿过文字、表格越界、底部溢出、字号过小、颜色含义漂移和空白失衡。发现问题时只修改相应页面的布局或措辞，并重新构建和渲染。

- [ ] **Step 3: 运行反向提纲与五维自审**

逐页记录一句主题句，确认它们按 Stage 1 → Stage 2 → Stage 3 → 独立验证 → 结果顺序连接。然后回答贡献、写作清晰度、实验强度、评测完整性和方法合理性五组问题；无法由现有实验回答的内容必须在第 15 页限制中承认，不新增主张。

- [ ] **Step 4: 运行最终验证**

```bash
PYTHONPATH=/tmp/codex-staged-report-tools /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest reports.staged_visual_search.test_report -v
git diff --check
git status --short
```

Expected: 测试全部通过，`git diff --check` 无输出，工作树只保留用户原有未跟踪实验材料。

- [ ] **Step 5: 提交必要的视觉修正**

仅在 Step 2–3 产生修改时提交：

```bash
git add reports/staged_visual_search
git commit -m "docs: polish staged method report"
```
