# Robust-Transfer Uncertainty Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one single-GPU-compatible, uncertainty-aware Stage 3 policy whose nested partition/source-group development evidence and one-shot MME-RealWorld-Lite evidence meet the approved robust-transfer gates.

**Architecture:** Preserve Stage 1/2 and the fixed six-branch Stage 3 observations. First make offline fitting traverse and meter exactly the same prefixes as runtime replay. Then add a focused robust-transfer selector that chooses one shared risk configuration by inner source-group OOF, evaluates the selection procedure with outer named-partition folds, and refits only after the nested gate passes. Add MME compatibility and the sealed external gate only after development succeeds.

**Tech Stack:** Python 3.11, NumPy, `unittest`, existing CVSearch evidence-gap replay, one NVIDIA H200 GPU for inference.

## Global Constraints

- Use `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11`.
- Use at most one visible GPU for every inference process.
- Treat `development` and `validation_v3` labels as opened development data.
- Never modify Stage 1/2 outputs or the fixed Stage 3 observation JSONL files.
- Preserve exact `FALLBACK_P0` and `STOP_P0` behavior.
- Keep action logic, features, risk penalty, and decision boundary shared; backbone/answer-type may select only a numeric calibration head.
- Do not use dataset name, ordinal, category, correctness, answer, or GT boxes in inference decisions.
- Require all eight cell deltas `>=0`, both backbone deltas `>0`, net gain `>=10/256`, corrections greater than corruptions, and mean observations `<=12.8`; prefer zero corruptions and mean observations `<=11.52`.
- Do not open MME-RealWorld-Lite outcomes until code, policy, manifests, commands, and gates are frozen and committed.

---

### Task 1: Make offline snapshots runtime-reachable and exactly metered

**Files:**
- Modify: `cvsearch/eval/replay_uncertainty_support.py`
- Modify: `cvsearch/eval/freeze_uncertainty_support.py`
- Test: `tests/test_replay_uncertainty_support.py`
- Test: `tests/test_freeze_uncertainty_support.py`

**Interfaces:**
- Produces: `CandidateSnapshot.observations: int`, the exact cumulative revealed-view count.
- Produces: `_RiskTopic.stop_observations: int`, the exact no-replacement replay cost.
- Preserves: `candidate_snapshots(...) -> tuple[CandidateSnapshot, ...]` and `replay_uncertainty_support(...) -> dict[str, Any]`.

- [ ] **Step 1: Write the unreachable-candidate and observation-count tests**

```python
def test_candidate_snapshots_stop_current_branch_after_unparseable_view(self):
    phase1, split = rescue_rows()
    branches = audit_value(split)["branches"]
    branches[0]["tight_view"]["answer"] = "not an option"
    branches[0]["context_view"]["answer"] = "B"
    snapshots = candidate_snapshots(phase1, split, calibration(), policy())
    self.assertFalse(any(item.branch_index == 0 for item in snapshots))

def test_snapshot_observations_equal_runtime_replacement_cost(self):
    phase1, split = rescue_rows()
    # Configure branch 1 as the first replaceable two-view candidate.
    snapshots = candidate_snapshots(phase1, split, calibration(), policy())
    decision = replay_uncertainty_support(phase1, split, calibration(), policy())
    selected = next(item for item in snapshots if item.output == decision["selected_output"])
    self.assertEqual(selected.observations, decision["observations"])
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest -v \
  tests.test_replay_uncertainty_support \
  tests.test_freeze_uncertainty_support
```

Expected: failure because branch-0 snapshots remain reachable and `CandidateSnapshot` lacks `observations`.

- [ ] **Step 3: Implement the minimal traversal and cost fix**

```python
@dataclass(frozen=True)
class CandidateSnapshot:
    branch_index: int
    revealed_roles: tuple[str, ...]
    observations: int
    # existing fields remain unchanged

# In candidate_snapshots:
for branch in prepared.branches:
    for revealed_count, view in enumerate(branch.views, start=1):
        observed_views.append(view)
        result.extend(_snapshots(..., observations=len(observed_views)))
        if view.canonical_answer is None:
            break
```

Use one no-replacement `replay_uncertainty_support` call per `_RiskTopic` to
obtain `stop_observations`; `_risk_topic_outcome` returns the selected
example's exact observation count or `topic.stop_observations`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all focused tests pass.

- [ ] **Step 5: Commit the reachability fix**

```bash
git add cvsearch/eval/replay_uncertainty_support.py \
  cvsearch/eval/freeze_uncertainty_support.py \
  tests/test_replay_uncertainty_support.py \
  tests/test_freeze_uncertainty_support.py
git commit -m "fix: align selector snapshots with runtime replay"
```

### Task 2: Define auditable robust-transfer gates and ranking

**Files:**
- Create: `cvsearch/eval/robust_transfer_selector.py`
- Create: `tests/test_robust_transfer_selector.py`

**Interfaces:**
- Produces: `AcceptanceCriteria(max_mean_observations=12.8, preferred_mean_observations=11.52, minimum_net_gain=10)`.
- Produces: `evaluate_acceptance(metrics: PolicyMetrics, topics: int) -> tuple[str, ...]`.
- Produces: `robust_rank(metrics: PolicyMetrics, topics: int, grid_index: int) -> tuple[Any, ...]`.

- [ ] **Step 1: Write failing gate and ordering tests**

```python
def test_gate_requires_positive_both_backbones_and_nonnegative_cells(self):
    failures = evaluate_acceptance(metrics(
        net_gain=12,
        backbone_deltas={"qwen": 0, "internvl": 12},
        cell_deltas=all_zero_cells(),
    ), topics=112)
    self.assertIn("qwen backbone is not strictly positive", failures)

def test_rank_prefers_larger_worst_backbone_before_pooled_gain(self):
    balanced = metrics(net_gain=10, backbone_deltas={"qwen": 4, "internvl": 6})
    concentrated = metrics(net_gain=20, backbone_deltas={"qwen": 1, "internvl": 19})
    self.assertLess(robust_rank(balanced, 112, 0), robust_rank(concentrated, 112, 1))

def test_gate_uses_actual_mean_observations(self):
    self.assertIn("mean observations exceed 12.8", evaluate_acceptance(
        metrics(observations=1434), topics=112,
    ))
```

- [ ] **Step 2: Run the new test module and verify RED**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest -v \
  tests.test_robust_transfer_selector
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement pure gate and ranking functions**

```python
@dataclass(frozen=True)
class AcceptanceCriteria:
    max_mean_observations: float = 12.8
    preferred_mean_observations: float = 11.52
    minimum_net_gain: int = 10

def robust_rank(metrics, topics, grid_index):
    backbones = dict(metrics.backbone_deltas)
    cells = dict(metrics.cell_deltas)
    mean = metrics.observations / topics
    return (
        metrics.corruptions != 0,
        min(backbones.values()) <= 0,
        min(cells.values()) < 0,
        metrics.net_gain < 10,
        mean > 12.8,
        -min(backbones.values()),
        -min(cells.values()),
        -metrics.net_gain,
        mean > 11.52,
        metrics.observations,
        grid_index,
    )
```

- [ ] **Step 4: Verify GREEN and commit**

Run the Step 2 command, then:

```bash
git add cvsearch/eval/robust_transfer_selector.py \
  tests/test_robust_transfer_selector.py
git commit -m "feat: define robust transfer acceptance gates"
```

### Task 3: Implement nested partition/source-group selection

**Files:**
- Modify: `cvsearch/eval/robust_transfer_selector.py`
- Modify: `tests/test_robust_transfer_selector.py`

**Interfaces:**
- Produces: `select_shared_configuration(records: Sequence[DevelopmentRecord]) -> SharedRiskSelection`.
- Produces: `nested_partition_validation(partitions: Mapping[str, Sequence[DevelopmentRecord]]) -> NestedTransferSelection`.
- Produces: `NestedTransferSelection.outer_folds`, `combined_oof_metrics`, `refit_policy`, and `failures`.

- [ ] **Step 1: Write failing leakage, shared-rule, and deterministic tests**

```python
def test_outer_partition_never_enters_train_groups(self):
    result = nested_partition_validation({"development": dev, "validation_v3": val})
    for fold in result.outer_folds:
        self.assertTrue(set(fold.train_groups).isdisjoint(fold.held_out_groups))

def test_penalty_and_boundary_are_shared_across_calibration_heads(self):
    result = select_shared_configuration(dev)
    values = {(head.risk_penalty, head.decision_boundary)
              for _, head in result.refit_calibrators}
    self.assertEqual(len(values), 1)

def test_nested_selection_is_byte_deterministic(self):
    first = nested_partition_validation(partitions).to_dict()
    second = nested_partition_validation(partitions).to_dict()
    self.assertEqual(first, second)
```

- [ ] **Step 2: Run and verify RED**

Run the Task 2 test command. Expected: missing selector functions.

- [ ] **Step 3: Implement cached inner OOF selection**

For every declared `(target_mode, degree, l2)` base, fit per-stratum heads with
deterministic four-fold source-group isolation. Apply one shared `(risk_penalty,
decision_boundary)` to every head. Aggregate all strata before applying
`evaluate_acceptance`; never select a stratum independently by its labels.

```python
for base in BASE_CONFIGURATIONS:
    oof_heads = fit_group_excluded_heads(topics, base)
    for penalty, boundary in SHARED_ACTION_GRID:
        metrics = score_all_topics(topics, bind(oof_heads, penalty, boundary))
        candidates.append((robust_rank(metrics, len(topics), grid_index), ...))
```

- [ ] **Step 4: Implement two-direction outer validation and full-pool refit**

```python
for held_out_name in sorted(partitions):
    train = concatenate(records for name, records in partitions.items()
                        if name != held_out_name)
    chosen = select_shared_configuration(train)
    outer = score_refit_on_holdout(chosen, partitions[held_out_name])
    folds.append(OuterFold(...))
combined = add_metrics([fold.metrics for fold in folds], all_records)
refit = select_shared_configuration(all_records)
```

- [ ] **Step 5: Verify focused tests and commit**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest -v \
  tests.test_robust_transfer_selector \
  tests.test_freeze_uncertainty_support \
  tests.test_replay_uncertainty_support
git add cvsearch/eval/robust_transfer_selector.py \
  tests/test_robust_transfer_selector.py
git commit -m "feat: select risk policy with nested grouped validation"
```

### Task 4: Add the robust-development CLI and authenticated artifacts

**Files:**
- Modify: `cvsearch/eval/score_uncertainty_support.py`
- Modify: `tests/test_score_uncertainty_support.py`
- Create: `reproduction/evidence_gap/adaptive_search_v10/.gitkeep`

**Interfaces:**
- Adds command: `freeze-robust-development`.
- Consumes: `--development-root`, `--stage2-root`, `--split-root`, and `--support-calibration`.
- Produces: `--policy-out` and `--report-out` with partition, source-group, policy, and decision hashes.

- [ ] **Step 1: Write a failing deterministic CLI test**

```python
arguments = [
    "freeze-robust-development",
    "--development-root", str(development),
    "--stage2-root", str(stage2),
    "--split-root", str(split),
    "--support-calibration", str(support),
    "--policy-out", str(policy_out),
    "--report-out", str(report_out),
]
self.assertEqual(main(arguments), 0)
self.assertEqual(json.loads(report_out.read_text())["data_scope"],
                 "opened_development_nested_oof")
```

- [ ] **Step 2: Run the CLI test and verify RED**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest -v \
  tests.test_score_uncertainty_support.SelectorCliTests
```

- [ ] **Step 3: Add the command using existing suite loaders and writers**

The command loads the original development observations plus the paired
`validation_v3` Stage-2/SPLIT observations, calls
`nested_partition_validation`, writes no evaluator labels into the policy,
and returns `2` with an exact failure list if the nested gate misses.

- [ ] **Step 4: Verify, commit, and run the real CPU replay**

```bash
RAW=/mnt/data3/data_xingrui/lueq/LOGIV_V2/.worktrees/stage3-adaptive-split-search/reproduction/evidence_gap/adaptive_search_v7/raw
PY=/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11
$PY -m cvsearch.eval.score_uncertainty_support freeze-robust-development \
  --development-root "$RAW/development" \
  --stage2-root "$RAW/validation_v3/stage2" \
  --split-root "$RAW/validation_v3/split" \
  --support-calibration reproduction/evidence_gap/adaptive_search_v7/split-calibration-manifest-v2.json \
  --policy-out reproduction/evidence_gap/adaptive_search_v10/robust-policy-v1.json \
  --report-out reproduction/evidence_gap/reports/robust-transfer-development-v1.json
```

If the command returns `2`, preserve the report, diagnose the failed fold/cell,
and change one declared calibration/search factor at a time with a new failing
test. Do not weaken acceptance gates.

```bash
git add cvsearch/eval/score_uncertainty_support.py \
  tests/test_score_uncertainty_support.py \
  reproduction/evidence_gap/adaptive_search_v10 \
  reproduction/evidence_gap/reports/robust-transfer-development-v1.json
git commit -m "exp: validate robust transfer selector on opened data"
```

### Task 5: Add label-blind five-option MME compatibility

Before Task 5, iterate Task 4 with the diagnosed robust-transfer v2 family:

- add a frozen aggregate evidence vector derived only from already observed
  render/support/canonical-answer values;
- fit one global standardized logistic benefit/harm head with grouped OOF;
- search one shared two-view confirmation rule and global observation budget;
- preserve legacy policy loading and exact P0 fallback;
- regenerate the real nested report and proceed only when every development
  gate passes.

This is a refinement of Tasks 3–4, not a new inference stage. It must use the
same red-green-refactor and authenticated-artifact procedure.

If aggregate scalar models fail exact nested replay, add the bounded pairwise
verifier described in the design: a shared observation-eight proposal, a
whole-image plus two-crop evidence sheet, two answer-order reversals, normalized
two-choice loss confidence, and exact P0 fallback. Generate the label-blind
verifier observations for Qwen and InternVL sequentially on one GPU. Select the
shared agreement/confidence thresholds only afterward with the same outer
partition and inner source-group isolation; verifier calls are charged as
observations.

The pairwise family failed its transfer gate and has been retained as negative
evidence. Continue with the candidate-free refinement from the design:

- [x] Bind the Qwen2.5-VL-32B checkpoint and 4,194,304-pixel budget into every
  record identity and collection manifest.
- [x] Collect both current partitions for both backbones and verify that every
  call succeeds.
- [x] Replay the fixed 0.6-confidence disagreement veto and 0.4/0.6 fallback
  agreement rule. Current result: `+18/512`, zero corruption, Qwen/InternVL
  `+9/+9`, all cells nonnegative, mean observations `9.04`.
- [ ] Collect the same candidate-free evidence on `validation_v1`,
  `validation_v2`, and `final_v2`, reusing an answer across backbones only
  under exact render/prompt/choice/model/budget hashes.
- [ ] Run outer-partition/inner-source-group selection with the verifier signal
  present in every training and held-out partition. Keep the label-exposed
  `+21/512` reachability scan diagnostic-only.
- [ ] Freeze only after the current two-partition OOF result reaches `+20/512`
  with both backbones positive, all eight cells nonnegative, corrections above
  corruptions, and mean observations no greater than 12.8.

**Files:**
- Modify: `cvsearch/evidence_gap/answers.py`
- Modify: `cvsearch/eval/replay_adaptive_search.py`
- Modify: `cvsearch/perform_EGSearch.py`
- Test: `tests/test_evidence_gap_answers.py`
- Test: `tests/test_perform_evidence_gap_search.py`

**Interfaces:**
- Changes: `official_letter(raw_output: str, allowed: str = "ABCD") -> str | None`.
- Changes: `aggregate_single_choice(raw_output: str, allowed: str = "ABCD") -> AnswerRecord`.
- Preserves `ABCD` as the default for all existing benchmarks.
- Uses `ABCDE` only when a sanitized option-single row proves it has five choices.

- [ ] **Step 1: Write failing A-E parsing and legacy fail-closed tests**

```python
self.assertEqual(official_letter("E", allowed="ABCDE"), "E")
self.assertIsNone(official_letter("E"))
self.assertEqual(aggregate_single_choice("E", allowed="ABCDE").canonical_answer, "E")
```

- [ ] **Step 2: Run and verify RED**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest -v \
  tests.test_evidence_gap_answers \
  tests.test_perform_evidence_gap_search
```

- [ ] **Step 3: Implement the optional allowed alphabet and strict MME output validation**

Validate that `allowed` is a unique uppercase contiguous prefix of `ABCDE`.
For `mme-realworld-lite`, require an option-single string whose canonical letter
is in `ABCDE`. Do not inspect the answer field.

- [ ] **Step 4: Verify and commit**

Run the Step 2 command and the three selector test modules, then commit with:

```bash
git add cvsearch/evidence_gap/answers.py \
  cvsearch/eval/replay_adaptive_search.py \
  cvsearch/perform_EGSearch.py \
  tests/test_evidence_gap_answers.py \
  tests/test_perform_evidence_gap_search.py
git commit -m "feat: support sealed five-option MME replay"
```

### Task 6: Freeze the one-GPU external manifest and run once

**Files:**
- Create: `reproduction/evidence_gap/adaptive_search_v10/mme-realworld-lite-manifest.json`
- Create: `reproduction/evidence_gap/reports/robust-transfer-mme-realworld-lite.json`
- Modify only if tests require it: `cvsearch/eval/score_uncertainty_support.py`
- Test: `tests/test_score_uncertainty_support.py`

**Interfaces:**
- Produces an authenticated manifest binding dataset, code, Qwen, InternVL,
  SAM, CLIP, spaCy, policy, calibration, commands, and `CUDA_VISIBLE_DEVICES`.
- Produces a report with per-backbone Direct/CVSearch/robust counts,
  corrections, corruptions, deltas, observations, hashes, and gate failures.

- [ ] **Step 1: Acquire and verify the official dataset before reading outcomes**

Download the official MME-RealWorld-Lite package into
`datasets/hr_data/mme-realworld-lite`, verify that every referenced image
exists, and record file hashes. Do not run the evaluator.

- [ ] **Step 2: Write and verify the manifest gate test**

```python
def test_external_gate_requires_each_backbone_nonnegative_and_pooled_positive(self):
    report = sealed_external_report(qwen_delta=0, internvl_delta=1)
    self.assertEqual(evaluate_external_gates(report)["failures"], [])
```

The gate additionally requires corrections greater than corruptions and every
hash/provenance audit to be true.

- [ ] **Step 3: Run label-blind one-sample smoke tests on one GPU**

Use `CUDA_VISIBLE_DEVICES=0` and the pinned Qwen/InternVL snapshot paths from
the prior launch manifests. Smoke output is checked for schema, transitions,
fallback, and hashes only; correctness remains unopened.

- [ ] **Step 4: Commit the frozen manifest before full inference**

```bash
git add reproduction/evidence_gap/adaptive_search_v10/mme-realworld-lite-manifest.json \
  tests/test_score_uncertainty_support.py cvsearch/eval/score_uncertainty_support.py
git commit -m "exp: freeze sealed MME robust transfer gate"
```

- [ ] **Step 5: Run Qwen then InternVL sequentially on one GPU and score once**

Monitor process liveness, JSONL growth, GPU memory, timeout, and logs. Never run
the two backbones concurrently. Generate label-blind decisions before invoking
the evaluator. Write the final report once and do not retune from it.

- [ ] **Step 6: Run full verification and commit passing artifacts**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest discover -s tests -v
git diff --check
```

Verify every requirement in the design against the development and external
reports. If and only if all gates pass, commit the report and mark the goal
complete. Otherwise retain the exact failure report; continue only with a new
unopened external source, never by tuning on MME outcomes.
