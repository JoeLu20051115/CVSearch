# Fixed-Pool SPLIT Ranking Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only evaluator that compares five rankings over the exact same sixteen SPLIT leaves and exhaustively decomposes frozen Stage 3b validation failures.

**Architecture:** Reconstruct visual and CLIP percentile components from frozen all-root audits and source pixels, score fixed leaf identities under deterministic policies, and join compact frozen decision rows to raw branch outputs for failure accounting. Keep all geometry and labels inside `cvsearch/eval`.

**Tech Stack:** Python 3, standard library, Pillow, existing CVSearch evaluator helpers, `unittest`.

## Global Constraints

- Do not change inference, candidate generation, selection, calibration, or frozen reports.
- Rank exactly sixteen unique depth-two patches for every eligible topic.
- Use only opened V*/TreeBench development geometry for ranking metrics.
- Require identical candidate geometry across backbones, but score each
  backbone-specific query ranking independently.
- Treat `validation_v3` only as a post-hoc failure report; never tune from it.
- Use exact random expectations rather than sampled random permutations.
- Add no dependency and do not require a GPU.

---

### Task 1: Lock fixed-pool reconstruction and ranking metrics

**Files:**
- Create: `tests/test_eval_split_ranking_ablation.py`
- Create: `cvsearch/eval/eval_split_ranking_ablation.py`

**Interfaces:**
- Produces: `evaluate_fixed_pool(observations_by_backbone, image_roots, ks=(1, 3, 6)) -> dict[str, Any]`.
- Produces: `exact_random_metrics(labels, ks) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests for exact pool identity and component recovery**

Construct one synthetic all-root audit with four roots and sixteen probes. Use
known visual values and combined scores, then assert the evaluator rejects a
missing/duplicate patch and recovers the known CLIP percentile within `1e-9`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_eval_split_ranking_ablation -v`

Expected: import failure because `cvsearch.eval.eval_split_ranking_ablation`
does not exist.

- [ ] **Step 3: Implement the minimum reconstruction helpers**

Reuse `SplitPatch`, `rank_split_children`, `_split_visual_features`, and the
existing V*/TreeBench geometry labelers. Validate exact v3 policy, four roots,
four probes per root, sixteen unique `(path, box)` identities, finite scores,
and identical backbone copies.

- [ ] **Step 4: Write failing tests for grid, visual, CLIP, combined, and exact random metrics**

Use a sixteen-label fixture with known first-hit ranks. Assert Recall@1/3/6,
pool recall, MRR, mean first rank, deterministic tie-breaking, and exact
hypergeometric random expectations.

- [ ] **Step 5: Implement fixed ranking and aggregate metrics**

Use the mean of root and child components for each leaf, sort descending with
path tie-breaking, and aggregate each backbone ranking after cross-backbone
candidate-geometry verification. Return aggregate, per-backbone, and
per-dataset reports plus explicit gates.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_eval_split_ranking_ablation -v`

Expected: all Task 1 tests pass with no warnings.

### Task 2: Add exhaustive frozen-failure decomposition

**Files:**
- Modify: `tests/test_eval_split_ranking_ablation.py`
- Modify: `cvsearch/eval/eval_split_ranking_ablation.py`

**Interfaces:**
- Produces: `decompose_frozen_failures(score_report, observations_by_cell) -> dict[str, Any]`.

- [ ] **Step 1: Write failing tests for every answer-level category**

Build compact decision rows covering `converted`, `selector_abstained`,
`selector_wrong_choice`, `no_correct_observed_answer`, and `corruption`.
Assert baseline-error categories are mutually exclusive and sum to the exact
Stage 2 error count.

- [ ] **Step 2: Run the focused category tests and verify RED**

Run the single new unittest by its fully qualified test name. Expected:
`decompose_frozen_failures` is missing.

- [ ] **Step 3: Implement answer-level decomposition using existing scorers**

Join rows by `cell` and `_eg_ordinal`, call `candidate_outputs` and
`official_correctness`, and classify each official unit without re-running the
selector. Validate that recomputed Stage 2/Stage 3b correctness and aggregate
corrections/corruptions match the frozen report.

- [ ] **Step 4: Write failing V*/TreeBench refinement tests**

Create cases for no sufficient leaf, sufficient leaf outside the six observed
branches, sufficient observed branch without a correct answer, and a converted
unit. Assert HR rows are never assigned a geometry category.

- [ ] **Step 5: Implement geometry refinement and reconciliation gates**

Use all sixteen probe boxes for pool sufficiency and the six tight branch
crops for observation sufficiency. Return aggregate, cell, backbone, and
dataset counts with exact reconciliation booleans.

- [ ] **Step 6: Run the focused suite and verify GREEN**

Run the test module. Expected: all Task 1 and Task 2 tests pass.

### Task 3: Add CLI, generate the bound report, and verify

**Files:**
- Modify: `tests/test_eval_split_ranking_ablation.py`
- Modify: `cvsearch/eval/eval_split_ranking_ablation.py`
- Create: `reproduction/evidence_gap/reports/split-ranking-ablation-v1.json`

**Interfaces:**
- CLI module: `python -m cvsearch.eval.eval_split_ranking_ablation`.

- [ ] **Step 1: Write a failing CLI/report-schema test**

Assert strict required arguments, canonical JSON output, source SHA-256
bindings, development/validation scope declarations, and an aggregate
`success` equal to the conjunction of all declared gates.

- [ ] **Step 2: Run the CLI test and verify RED**

Expected: CLI entry point or report schema is missing.

- [ ] **Step 3: Implement the smallest explicit CLI**

Accept the development observation root, V*/TreeBench image roots, frozen
validation score report, frozen validation split root, and output path. Load
only the declared files, compute their SHA-256 digests, and write canonical
JSON atomically.

- [ ] **Step 4: Run focused tests and the real evaluator**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_eval_split_ranking_ablation -v
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m cvsearch.eval.eval_split_ranking_ablation \
  --development-root reproduction/evidence_gap/adaptive_search_v7/raw/development \
  --vstar-image-root datasets/hr_data/vstar \
  --treebench-image-root datasets/hr_data/treebench \
  --validation-report reproduction/evidence_gap/reports/stage3b-validation-v3.json \
  --validation-split-root reproduction/evidence_gap/adaptive_search_v7/raw/validation_v3/split \
  --output reproduction/evidence_gap/reports/split-ranking-ablation-v1.json
```

Expected: report `success` is true, all sixteen-patch/cross-backbone/hash and
failure-accounting gates pass, and combined ranking passes the predeclared
Recall@3/Recall@6/MRR gates.

- [ ] **Step 5: Run regression tests**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest \
  tests.test_eval_split_ranking_ablation \
  tests.test_freeze_split_calibration \
  tests.test_analyze_split_search \
  tests.test_evidence_gap_split_search \
  tests.test_evidence_gap_split_observation \
  tests.test_replay_split_search -v
```

Expected: all tests pass, with no new warning or failure.

- [ ] **Step 6: Audit and commit**

Verify the report contains no source answers, target boxes, or raw model
outputs; inspect `git diff --check` and `git status`; commit only evaluator,
tests, spec, plan, and compact report.
