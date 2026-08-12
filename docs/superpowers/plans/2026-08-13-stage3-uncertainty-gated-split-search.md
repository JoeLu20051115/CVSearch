# Stage 3 Uncertainty-Gated Split Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Add bounded two-depth SPLIT/BACKTRACK evidence generation after the
frozen Stage 1 v6 and Stage 2 v7 pipeline, then demonstrate non-regressing
accuracy in every tested backbone/dataset cell and positive pooled accuracy for
both backbones and all four datasets.

**Architecture:** Stage 3 is an append-only candidate layer. A pure inference
module generates native-coordinate 2-by-2 children, ranks them with query
relevance plus visual information, and describes support trajectories. Runtime
observation records tight/context views but cannot replace Stage 2. A separate
label-blind replay selector authenticates Stage 1/2 identity and admits only
two-view-confirmed answers; evaluator-only code opens labels after decisions
are frozen.

**Tech Stack:** Python 3.11 standard library, NumPy/PIL already in the
repository, `unittest`, Qwen2.5-VL-7B, InternVL2.5-8B, CLIP-L/14, SAM 3,
JSON/JSONL manifests.

## Global Constraints

- Keep every tracked Stage 1 v6 and Stage 2 v7 artifact byte-identical.
- Do not route on backbone, benchmark, resolution, category, ordinal, answer,
  correctness, or target boxes during inference.
- Do not replace an answer inside `perform_EGSearch`; observation is
  candidate-only and exact Stage 2 fallback is mandatory.
- Use fixed 2-by-2 children, 12.5% overlap, depth at most two, and at most two
  observed branches. No artificial super-resolution or unbounded queue.
- Require two distinct agreeing rendered views, calibrated sufficient support,
  positive support gain, and a non-decreasing trajectory before replacement.
- Freeze disjoint validation and final-holdout manifests before the first new
  Stage 3 model inference. Development labels may tune only a shared threshold
  and shared minimum gain; validation and final labels may not tune policy.
- Preserve raw experiments outside git. Commit code, tests, configs, manifests,
  and compact reports only.

---

### Task 1: Pure split geometry, ranking, and trajectory confirmation

**Files:**
- Create: `cvsearch/evidence_gap/split_search.py`
- Create: `tests/test_evidence_gap_split_search.py`

**Interfaces:**
- `SplitPatch(path, box)` and `RankedSplitPatch`
- `generate_split_children(parent: SplitPatch) -> tuple[SplitPatch, ...]`
- `rank_split_children(patches, relevance, edge_density, feature_deviation)`
- `mann_kendall_s(values: Sequence[float]) -> int`
- `confirm_split_branch(...) -> SplitConfirmation`

- [ ] **Step 1: Write failing geometry and parity tests**

Cover odd/even boxes, clipping, stable path identities, duplicate rejection,
two recursive depths, and exact parity with the historical Phase 12 child
helper. Assert the production module never imports `cvsearch.eval`.

- [ ] **Step 2: Verify RED**

Run:

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_split_search -v
```

Expected: import failure for the missing production module.

- [ ] **Step 3: Implement fixed child geometry and shared ranking**

Implement the frozen 2-by-2/12.5%-overlap contract locally. Rank all four
siblings by 0.70 query-relevance percentile plus 0.30 visual-information
percentile, with visual information equally weighting edge density and feature
deviation. Preserve deterministic path order for ties.

- [ ] **Step 4: Add failing trajectory and fail-closed tests**

Test Mann-Kendall S for rising, falling, plateaued, and nonfinite sequences.
Test confirmation rejection for equal hashes, disagreeing answers, decreasing
support, insufficient final support, nonpositive gain, malformed answers, and
equally strong conflict. Test component-safe HR confirmation.

- [ ] **Step 5: Implement the smallest immutable confirmation result**

Return only `confirmed`, `canonical_answer`, `reason`, `trajectory_s`,
`support_gain`, and `selection_score`. Accept no evaluator fields and use no
benchmark-specific logic.

- [ ] **Step 6: Verify GREEN and commit**

Run the focused test and:

```text
git commit -m "feat: add bounded split search primitives"
```

---

### Task 2: Strict split observation data contract

**Files:**
- Modify: `cvsearch/evidence_gap/types.py`
- Modify: `tests/test_evidence_gap_types.py`
- Create: `tests/test_evidence_gap_split_observation.py`

**Interfaces:**
- `SplitViewObservation`
- `SplitBranchObservation`
- `SplitSearchAudit`
- optional `StepTrace.split_search_audit`

- [ ] **Step 1: Write failing exact-schema tests**

Construct valid tight/context views and two branches. Mutate every field in
turn to test finite supports, integer nested boxes, distinct render hashes,
unchanged P0 anchor, exact rank/query hashes, maximum depth/branch counts,
monotone budget ledgers, and immutable `to_dict()` snapshots.

- [ ] **Step 2: Verify RED**

Run:

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_types tests.test_evidence_gap_split_observation -v
```

Expected: imports fail for the new DTOs.

- [ ] **Step 3: Implement DTOs without changing legacy serialization**

Follow the existing Zoom/Expand immutable-audit pattern. Omit
`split_search_audit` from legacy `StepTrace.to_dict()` output when it is absent,
so all Stage 1/2 bytes remain unchanged.

- [ ] **Step 4: Verify serialization compatibility and commit**

Run the focused tests plus `tests.test_evidence_gap_zoom_observation` and
`tests.test_evidence_gap_expand_observation`, then commit:

```text
git commit -m "feat: define split observation audit contract"
```

---

### Task 3: Candidate-only runtime SPLIT observation

**Files:**
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `cvsearch/perform_EGSearch.py`
- Modify: `tests/test_evidence_gap_method.py`
- Modify: `tests/test_evidence_gap_split_observation.py`
- Create: `reproduction/evidence_gap/configs/dev_adaptive_ranking_observe_split_v1.json`

**Interfaces:**
- config keys `p5a_split_enabled`, `p5a_split_replacement_enabled`,
  `p5a_split_render_policy`, `p5a_split_max_observed_branches`
- render policy `native_2x2_overlap_two_scale_depth2_v1`

- [ ] **Step 1: Write failing all-or-none config tests**

Require the exact Stage 1 v6/Stage 2 v7 ranking and observation fields, disabled
replacement, the frozen render policy, and maximum two branches. Reject partial
groups, enabled legacy `enable_split`, larger search budgets, and incompatible
ranking profiles.

- [ ] **Step 2: Write a failing fake-runtime preservation test**

Use a deterministic image and fake model to observe four children, answer only
the top two at tight/context scales, and recurse once when the first branch is
insufficient. Assert exact Stage 2 output/rank bytes, append-only trace data,
native crop coordinates, two distinct render hashes, call/pixel accounting,
one backtrack, and fail-closed handling for render/model/budget failures.

- [ ] **Step 3: Verify RED**

Run:

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_method tests.test_evidence_gap_split_observation -v
```

Expected: unknown config fields and absent SPLIT audit.

- [ ] **Step 4: Implement candidate observation only**

Reuse existing support prompts, answer adapters, query relevance, visual
feature extraction, render hashing, and budget ledger. Add no new model calls
outside admitted SPLIT branches. Never invoke replacement in the runtime.

- [ ] **Step 5: Verify GREEN, one-row dual-backbone smoke test, and commit**

Run focused observation/adapter tests, then one Qwen and one InternVL row. Both
must produce valid audits without traceback or OOM. Commit:

```text
git commit -m "feat: observe two-depth split candidates"
```

---

### Task 4: Label-blind trajectory replay and evaluator

**Files:**
- Create: `cvsearch/eval/replay_split_search.py`
- Create: `cvsearch/eval/score_stage3_transfer.py`
- Create: `tests/test_replay_split_search.py`
- Create: `tests/test_score_stage3_transfer.py`

**Interfaces:**
- `select_split_candidate(stage2_row, split_row, calibration, policy)`
- `evaluate_stage3_rows(rows)`
- `split_candidate_oracle(rows)`

- [ ] **Step 1: Write failing authentication and selection tests**

Test exact Stage 1 rank and P0 identity, Stage 2 selected-output identity,
calibrator provenance, two-view agreement, trajectory direction, backtracking,
conflict retention, HR component projection, TreeBench single-choice parsing,
and rejection of label/correctness/target-box fields.

- [ ] **Step 2: Verify RED**

Run:

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_split_search tests.test_score_stage3_transfer -v
```

Expected: import failures for both new modules.

- [ ] **Step 3: Implement replay before scoring**

Apply the frozen backbone support calibrator, but one shared final-support
threshold and minimum-gain pair. Inspect ranked branches in order, backtrack
once after a negative/insufficient trajectory, stop on first confirmed answer,
and otherwise return the exact Stage 2 output.

- [ ] **Step 4: Implement evaluator-only paired metrics**

Report candidate-oracle gain, official accuracy, corrections/corruptions,
false replacements, per-cell/backbone/dataset aggregates, Brier/ECE/AUROC,
trajectory classes, depth/branch/backtrack counts, calls, pixels, and latency.
Labels are accepted only here after selection records are frozen.

- [ ] **Step 5: Verify GREEN and commit**

Run focused replay/evaluator tests and commit:

```text
git commit -m "feat: replay and score confirmed split search"
```

---

### Task 5: Freeze partitions and generate new candidates

**Files:**
- Create: `reproduction/evidence_gap/adaptive_search_v6/partition-manifest.json`
- Create: `reproduction/evidence_gap/adaptive_search_v6/calibration-manifest.json`
- Produce untracked: `reproduction/evidence_gap/adaptive_search_v6/raw/`

**Interfaces:**
- consumes exact annotations/checkpoints and frozen Stage 1/2 rows
- produces disjoint development, validation, and final-holdout identities

- [ ] **Step 1: Freeze manifests before model inference**

Hash annotations and baseline rows. Mark the previously exposed Stage 2 subsets
as development. Select deterministic disjoint validation and final ordinals
from remaining examples using SHA-256 of source identity plus seed `260813`;
do not inspect labels while selecting.

- [ ] **Step 2: Run candidate-oracle screening on development**

Observe SPLIT candidates on Qwen and InternVL for V*, HR-Bench 4K, HR-Bench 8K,
and TreeBench. Before selector tuning, require at least one newly correct
candidate for every backbone and every dataset. If a group has zero gain,
diagnose focus-patch miss, child-ranking miss, or insufficient depth/rendering
and change candidate generation only on development.

- [ ] **Step 3: Freeze the shared selector**

Search only the declared shared grids:

```text
final_support_threshold = (0.50, 0.60, 0.70, 0.80)
minimum_support_gain = (0.00, 0.05, 0.10, 0.20)
```

Select lexicographically by zero corruptions, maximum corrections, lower false
replacement rate, then lower cost. Freeze thresholds, calibrator hashes,
decision hashes, and source rows before opening validation labels.

- [ ] **Step 4: Run disjoint validation once**

Require every one of eight cells to be non-regressing, each backbone-pooled
accuracy to improve, and each dataset-pooled accuracy to improve. A failed
validation changes the design and requires a new untouched validation split;
it cannot tune thresholds on the failed labels.

- [ ] **Step 5: Commit compact frozen artifacts**

Commit manifests and compact diagnostics, excluding raw model outputs:

```text
git commit -m "exp: freeze stage3 split search policy"
```

---

### Task 6: Final holdout, full verification, and main integration

**Files:**
- Create: `reproduction/evidence_gap/reports/stage3-split-search-transfer.json`
- Verify all changed files and compact artifacts

- [ ] **Step 1: Open the final holdout once**

Run the frozen policy on both backbones and all four datasets. Record all eight
cell results and pooled metrics. If compute permits after gates pass, run the
complete benchmark partitions with the same frozen policy.

- [ ] **Step 2: Apply promotion gates**

Require byte-identical Stage 1/2 artifacts, exact rank/P0/Stage 2 identity,
zero regressing cells, positive accuracy for both backbone aggregates and all
four dataset aggregates, corrections greater than corruptions, calibration
no-harm, two-view evidence for every change, and bounded cost.

- [ ] **Step 3: Run focused and complete test suites**

```text
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest \
  tests.test_evidence_gap_split_search \
  tests.test_evidence_gap_split_observation \
  tests.test_replay_split_search \
  tests.test_score_stage3_transfer -v
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -q
```

Expected: zero failures/errors and only declared artifact-availability skips.

- [ ] **Step 4: Verify provenance, raw logs, and immutable hashes**

Run JSON/JSONL parsing, SHA-256 checks, `git diff --check`, and scan raw logs for
tracebacks, OOMs, NaNs, and incomplete jobs. Compare frozen Stage 1/2 hashes to
the committed v6/v7 reports.

- [ ] **Step 5: Commit final report and integrate**

Commit the compact report, merge `feature/stage3-adaptive-split-search` into
`main` without discarding user files, rerun the full test suite on `main`, and
report exact commits, per-cell deltas, calibration, and cost.
