# Cross-Backbone Phase 1 and Phase 2 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote a Phase 1 ranking rule that transfers across Qwen and InternVL, then complete Phase 2 with no-harm cross-validated calibration and a TreeBench dataset gate.

**Architecture:** Preserve Phase 1 v1 as historical evidence and add one context-conditioned visual-fusion coefficient for v2. Preserve the Phase 2 action policy and add a source-grouped selector that shrinks isotonic predictions toward raw support. Extend the existing observation DTOs only enough to carry TreeBench's existing `option_single` answer contract.

**Tech Stack:** Python 3.11 standard library, NumPy/PIL already in the repository, `unittest`, Qwen2.5-VL-7B, InternVL2.5-8B, CLIP-L/14, SAM 3, JSON/JSONL launch manifests.

## Global Constraints

- Do not alter the bytes of the Phase 1 v1 query profile, v1 config, or v1 report.
- Do not use backbone, benchmark, evaluator category, target boxes, answers, or correctness in inference routing.
- Do not add dependencies or benchmark-specific action thresholds.
- Keep every candidate; Top-3 is an evaluation budget, not pruning.
- Regenerate Phase 2 observations against Phase 1 v2 before making the final claim.
- Tune only on declared calibration/development rows; inspect holdout metrics only after freezing each rule.
- Preserve raw experiments outside git and commit only code, configs, manifests, and compact reports.

---

### Task 1: Phase 1 v2 context-conditioned visual fusion

**Files:**
- Modify: `cvsearch/evidence_gap/ranking.py`
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `cvsearch/eval/replay_vstar_ranking.py`
- Modify: `tests/test_evidence_gap_ranking.py`
- Modify: `tests/test_evidence_gap_method.py`
- Modify: `tests/test_replay_vstar_ranking.py`
- Create: `reproduction/evidence_gap/configs/dev_adaptive_ranking_v2.json`
- Create: `reproduction/evidence_gap/configs/dev_adaptive_ranking_observe_v3.json`

**Interfaces:**
- `QueryAwareNodeRanker(..., context_visual_discount: float | None = None)`
- v2 trace field `effective_visual_lambda: float`
- replay config key `context_visual_discount: float`

- [ ] **Step 1: Write failing ranker tests**

Add tests asserting that omission of the new argument preserves the exact v1
detail dictionaries, while `context_visual_discount=1.0` computes:

```python
effective_visual_lambda = max(
    0.0,
    min(1.0, visual_lambda - context_visual_discount * profile.context_demand),
)
```

The test must use real PIL crops where complexity and edge density prefer
different candidates, prove a relation query switches to the edge ordering,
and prove a detail query retains the feature-deviation ordering.

- [ ] **Step 2: Verify RED**

Run:

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_ranking tests.test_evidence_gap_method tests.test_replay_vstar_ranking -v
```

Expected: failures for the unknown constructor/config/replay key.

- [ ] **Step 3: Implement the minimal optional coefficient**

Validate the optional coefficient in `[0, 1]`, compute one effective visual
weight per rank event, and pass it to the existing `fuse_scores`. Add
`effective_visual_lambda` only when the v2 option is supplied, so v1 traces do
not gain a new field. Add one strict optional config key that is legal only with
the existing complete adaptive-ranking key pair and `query_linear` mode.

- [ ] **Step 4: Add replay parity and configs**

Make replay use the same effective visual formula. Copy the v1 configs without
changing unrelated fields; set `config_id` to `adaptive-ranking-v2` /
`adaptive-ranking-observe-v3` and set `context_visual_discount` to `1.0`.

- [ ] **Step 5: Verify GREEN, v1 bytes, and commit**

Run the focused tests, compare v1 tracked-file SHA-256 values against
`c1ee03a`, run CPU replay on Qwen and InternVL development traces, and commit:

```text
git commit -m "feat: make Phase-1 visual fusion context adaptive"
```

---

### Task 2: Source-grouped no-harm calibration selection

**Files:**
- Modify: `cvsearch/eval/replay_adaptive_search.py`
- Modify: `tests/test_replay_adaptive_search.py`

**Interfaces:**
- `FrozenSelectedCalibration.predict(raw_support) -> float`
- `freeze_selected_calibration(rows) -> FrozenSelectedCalibration`
- exact input row keys: `row_id`, `source_group`, `raw_support`, `support_sufficient`

- [ ] **Step 1: Write failing selection tests**

Use one synthetic grouped sample where raw support has the lowest leave-one-group-out
Brier score and assert selected weight `0.0` returns exact raw probabilities.
Use a second sample where monotone pooling is beneficial and assert a positive
weight is selected. Assert input rows containing `answer`, `correct`, or missing
`source_group` are rejected, and the legacy `freeze_isotonic_calibration`
manifest remains byte-identical.

- [ ] **Step 2: Verify RED**

Run:

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_adaptive_search -v
```

Expected: import failure for the selected calibration API.

- [ ] **Step 3: Implement grouped cross-validation**

For weights `(0.0, 0.125, 0.25, 0.5, 0.75, 1.0)`, leave one source group out,
fit the existing isotonic calibrator on the remainder, and predict:

```python
isotonic_with_tie = (1.0 - 1.0 / (n + 1)) * isotonic + raw / (n + 1)
prediction = (1.0 - weight) * raw + weight * isotonic_with_tie
```

Select by `(brier, ece_10, weight)`, fit on all calibration rows, and hash the
exact rows, fold metrics, selected weight, and final knots. Accept either legacy
or selected calibration in replay without changing action thresholds.

- [ ] **Step 4: Verify Qwen/InternVL selection and commit**

Reconstruct source-grouped calibration rows from frozen V* observations.
Expected selections are Qwen `0.75` and InternVL `0.0`; held-out Qwen Brier/ECE
must improve and InternVL must be equal to raw. Run focused tests and commit:

```text
git commit -m "feat: select no-harm support calibration"
```

---

### Task 3: TreeBench `option_single` observation contract

**Files:**
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `cvsearch/evidence_gap/types.py`
- Modify: `cvsearch/eval/replay_adaptive_search.py`
- Modify: `cvsearch/perform_EGSearch.py`
- Modify: `tests/test_evidence_gap_combined_observation.py`
- Modify: `tests/test_replay_adaptive_search.py`
- Create: `tests/test_perform_evidence_gap_treebench.py`
- Create: `tests/test_treebench_support_proxy.py`
- Create: `cvsearch/eval/treebench_support_proxy.py`

**Interfaces:**
- candidate answer payload for `option_single`: one raw string
- canonical answer: `official_letter(raw_output)`
- evaluator-only `treebench_geometry_support_label(annotation, crops) -> int`

- [ ] **Step 1: Write failing end-to-end single-choice tests**

Test a fake runtime that renders current/candidate views, emits normalized
support, answers one formatted TreeBench prompt, serializes a one-call batch
plan, replays the candidate through the official letter parser, and falls back
to P0 on malformed output. Assert the controller input contains no TreeBench
answer, category, index, or target boxes.

- [ ] **Step 2: Verify RED**

Run:

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_combined_observation tests.test_replay_adaptive_search tests.test_perform_evidence_gap_treebench tests.test_treebench_support_proxy -v
```

Expected: the observation preflight rejects `option_single` and the proxy module
is absent.

- [ ] **Step 3: Implement the narrow answer contract**

Reuse CVSearch's prompt text:

```python
question_input = question + "\n" + options + "\nAnswer with the option's letter from the given choices directly."
```

Charge one candidate-answer call, require a string payload, canonicalize only
inside replay, and extend exact DTO validation for a single answer hash. Do not
change V* or HR serialization.

- [ ] **Step 4: Implement evaluator-only geometry proxy**

Parse `target_instances` with `json.loads`, validate XYXY boxes, require all
targets for reasoning categories and per-target detail visibility for perception
categories, and keep this module under `cvsearch/eval/` so inference cannot
import it.

- [ ] **Step 5: Verify GREEN, run a one-row real smoke test, and commit**

Run focused tests and one Qwen plus one InternVL TreeBench observation row.
Require valid launch manifests, parseable letters, and no traceback/OOM. Commit:

```text
git commit -m "feat: observe TreeBench scale evidence"
```

---

### Task 4: Regenerate paired Phase 1/2 experiments

**Files:**
- Produce untracked: `reproduction/evidence_gap/adaptive_search_v3/raw/`
- Create: `reproduction/evidence_gap/reports/phase1-v2-transfer.json`
- Create: `reproduction/evidence_gap/reports/adaptive-search-v3-transfer.json`
- Create: `reproduction/evidence_gap/adaptive_search_v3/calibration-manifest.json`

**Interfaces:**
- consumes exact Qwen/InternVL checkpoints, V*/HR/TreeBench annotations, v2/v3 configs
- produces paired baseline/observation rows and compact frozen reports

- [ ] **Step 1: Freeze partitions before inference**

Record the Qwen 30-topic V* regression set, a disjoint InternVL spatial holdout,
the existing source-grouped HR subsets, and a deterministic TreeBench dev/holdout
subset. Store ordinal lists and annotation hashes before reading outcomes.

- [ ] **Step 2: Run Phase 1 v2 baselines**

Use `cvsearch.perform_EGSearch` with the v2 config on Qwen and InternVL. Evaluate
actual answers and ranking recall against original CVSearch and v1. Abort Phase
2 promotion if any Phase 1 acceptance gate fails.

- [ ] **Step 3: Run v3 candidate observations**

Use the observation v3 config on the same baseline rows for V*, HR-4K/8K, and
TreeBench. Parallelize only independent GPU jobs, preserve partial files, and do
not reuse v1-ranked observations as v2 evidence.

- [ ] **Step 4: Freeze calibration, decisions, and reports**

Fit only the grouped V* calibration partitions, freeze selected weights and
hashes, replay the shared policy on all datasets, then open evaluator labels.
Report answers, calibration, support AUROC, false stops, action gains, rank
identity, calls/pixels/latency, and per-cell deltas.

- [ ] **Step 5: Apply gates and commit compact artifacts**

Require no Phase 1 category regression, no Phase 2 cell regression, macro answer
gain, cross-backbone calibration no-harm, and zero rank drift. Commit reports,
configs, and manifests only.

---

### Task 5: Final integrity and regression verification

**Files:**
- Verify all changed and generated compact files

**Interfaces:**
- produces final reproducibility and integration evidence

- [ ] **Step 1: Run focused test suites**

Run ranking, replay, adaptive controller, observation, adapter, runner, V*/HR,
and TreeBench evaluator tests.

- [ ] **Step 2: Run the complete unit suite**

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -q
```

Expected: zero failures or errors; only declared artifact-availability skips.

- [ ] **Step 3: Verify integrity**

Run `git diff --check`, parse every committed JSON file with `python -m json.tool`,
verify report input hashes, scan all new logs for traceback/OOM, and compare v1
immutable file hashes to `c1ee03a`.

- [ ] **Step 4: Review the final diff and commit**

Every changed line must map to the design. Keep downloaded datasets, raw JSONL,
logs, symlinks, checkpoints, and temporary archives untracked. Commit final
verification metadata only if it adds evidence not already in the reports.
