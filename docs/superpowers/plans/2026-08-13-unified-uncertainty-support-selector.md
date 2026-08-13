# Unified Uncertainty-Support Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Stage 3b selector's scattered thresholds with one frozen uncertainty-support state machine that improves over CVSearch while passing the locked 112-topic, 256-unit regression gate.

**Architecture:** Keep the recorded sixteen-patch geometry, CLIP-plus-visual ordering, six branches, model answers, and existing support calibration immutable. Add an evaluator-only replay layer that converts five candidate-vs-P0 features into one monotone-calibrated utility, uses that value for every state transition, freezes one profile and threshold by source-grouped development OOF, and then performs a label-blind locked replay before scoring.

**Tech Stack:** Python 3.10+, standard library, existing `cvsearch.evidence_gap` answer/support helpers, `unittest`, deterministic canonical JSON and SHA-256 artifacts; no GPU and no new dependencies.

## Global Constraints

- Preserve exactly sixteen answer-free depth-two probes and the frozen `0.7 * CLIP + 0.3 * visual` ranking.
- Preserve the existing six answer-bearing branches, fixed branch order, rendered views, answers, raw supports, and per-backbone support calibrators.
- Inference must not receive benchmark names, evaluator categories, target boxes, correct answers, or correctness bits.
- A changed candidate needs at least two agreeing distinct render hashes, valid provenance, parseable canonical answers, and the existing raw-support floor.
- Use only `balanced`, `support_heavy`, and `uncertainty_light` profiles and thresholds `(0.00, 0.05, 0.10, 0.15, 0.20, 0.25)`.
- Group development OOF by underlying source across backbones, and group the same HR-Bench ordinal across 4K and 8K.
- Treat `validation_v3` as a locked regression only; its labels must not affect profiles, calibration, thresholds, features, or transitions.
- Fail closed to the exact Stage-2/P0 output on schema, provenance, calibration, accounting, or gate failure.
- Do not regenerate model observations and do not require a GPU.

---

## File map

- Create `cvsearch/eval/replay_uncertainty_support.py`: feature validation, continuous isotonic calibration, frozen policy parsing, candidate snapshots, and label-blind state-machine replay.
- Create `cvsearch/eval/freeze_uncertainty_support.py`: development source groups, grouped OOF policy selection, all-development refit, and frozen-policy serialization.
- Create `cvsearch/eval/score_uncertainty_support.py`: raw-row loading/alignment, decision-file generation, paired official-unit accounting, gate evaluation, and CLI entry points.
- Create `tests/test_replay_uncertainty_support.py`: calibrator, features, structural safety, transitions, fallback, and determinism.
- Create `tests/test_freeze_uncertainty_support.py`: source grouping, fold isolation, feasibility, tie-breaking, refit, and label-free policy payload.
- Create `tests/test_score_uncertainty_support.py`: label-blind decision boundary, exact accounting, hashes, gates, and compact reports.
- Create `reproduction/evidence_gap/adaptive_search_v8/frozen-policy-v1.json`: policy frozen exclusively from opened development.
- Create `reproduction/evidence_gap/adaptive_search_v8/locked-decisions-v1.json`: validation decisions generated without labels.
- Create `reproduction/evidence_gap/reports/unified-uncertainty-support-development-v1.json`: grouped-OOF and all-development audit.
- Create `reproduction/evidence_gap/reports/unified-uncertainty-support-locked-regression-v1.json`: final CVSearch-vs-selector regression table and gates.

### Task 1: Continuous utility calibration and unified feature score

**Files:**
- Create: `cvsearch/eval/replay_uncertainty_support.py`
- Create: `tests/test_replay_uncertainty_support.py`

**Interfaces:**
- Consumes: existing `FrozenSelectedCalibration` support predictions and finite numeric feature inputs.
- Produces: `AdvantageFeatures`, `UtilityIsotonicCalibrator.predict(score: float) -> float`, `fit_utility_isotonic(samples: Sequence[tuple[float, float]]) -> UtilityIsotonicCalibrator`, and `raw_advantage(features: AdvantageFeatures, weights: Sequence[float]) -> float`.

- [ ] **Step 1: Write failing calibrator and feature tests**

```python
class UtilityPrimitiveTests(unittest.TestCase):
    def test_pava_accepts_continuous_targets_and_is_monotone(self):
        fitted = fit_utility_isotonic(((0.1, 0.75), (0.2, 0.25), (0.3, 1.0)))
        predictions = [fitted.predict(value) for value in (0.1, 0.2, 0.3)]
        self.assertEqual(predictions, sorted(predictions))
        self.assertEqual(predictions, [0.5, 0.5, 1.0])

    def test_raw_advantage_is_the_only_weighted_numeric_score(self):
        features = AdvantageFeatures(0.8, 0.5, 0.7, 0.6, 0.4)
        self.assertAlmostEqual(raw_advantage(features, PROFILES["balanced"]), 0.6)

    def test_invalid_feature_or_profile_fails_closed(self):
        with self.assertRaises(ValueError):
            AdvantageFeatures(1.1, 0.5, 0.5, 0.5, 0.5)
        with self.assertRaises(ValueError):
            raw_advantage(AdvantageFeatures(0.5, 0.5, 0.5, 0.5, 0.5), (1.0, 0, 0, 0, 0.1))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_uncertainty_support -v`

Expected: `ModuleNotFoundError` for `cvsearch.eval.replay_uncertainty_support`.

- [ ] **Step 3: Implement immutable features and continuous PAVA**

```python
PROFILES = {
    "balanced": (0.20, 0.20, 0.20, 0.20, 0.20),
    "support_heavy": (0.10, 0.15, 0.35, 0.20, 0.20),
    "uncertainty_light": (0.10, 0.25, 0.25, 0.20, 0.20),
}
THRESHOLDS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)

@dataclass(frozen=True)
class AdvantageFeatures:
    uncertainty: float
    agreement: float
    support: float
    support_gain_01: float
    conflict_margin_01: float

    def __post_init__(self) -> None:
        values = astuple(self)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("advantage features must be finite values in [0, 1]")

@dataclass(frozen=True)
class UtilityIsotonicCalibrator:
    upper_bounds: tuple[float, ...]
    utilities: tuple[float, ...]

    def predict(self, score: float) -> float:
        if not math.isfinite(score) or not self.upper_bounds or len(self.upper_bounds) != len(self.utilities):
            raise ValueError("utility calibrator is invalid")
        return self.utilities[min(bisect_left(self.upper_bounds, score), len(self.utilities) - 1)]
```

Implement deterministic weighted PAVA by sorting `(score, target)` pairs, averaging equal scores, and merging adjacent blocks while the previous mean exceeds the next mean. Reject empty samples, non-finite scores, and targets outside `[0, 1]`.

- [ ] **Step 4: Run focused tests and the existing controller tests**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_uncertainty_support tests.test_adaptive_controller -v`

Expected: all tests pass, demonstrating that the new continuous calibrator does not alter the existing binary support calibrator.

- [ ] **Step 5: Commit the tested primitive**

```bash
git add cvsearch/eval/replay_uncertainty_support.py tests/test_replay_uncertainty_support.py
git commit -m "feat: add unified uncertainty support utility"
```

### Task 2: Label-blind candidate snapshots and state machine

**Files:**
- Modify: `cvsearch/eval/replay_uncertainty_support.py`
- Modify: `tests/test_replay_uncertainty_support.py`

**Interfaces:**
- Consumes: `stage2_row: Mapping[str, Any]`, `split_row: Mapping[str, Any]`, a `FrozenSelectedCalibration`, and `UnifiedPolicy`.
- Produces: `candidate_snapshots(...) -> tuple[CandidateSnapshot, ...]` and `replay_uncertainty_support(...) -> dict[str, Any]` with `selected_output`, `stage2_selected_output`, `selected_source`, `reason`, `selected_branch`, `observations`, and `transitions`.

- [ ] **Step 1: Add failing transition and safety tests with synthetic rows**

```python
def test_two_distinct_agreeing_views_replace_p0(self):
    split = split_row(branches=[branch("B", "B", raw_supports=(0.9, 0.8))])
    decision = replay_uncertainty_support(stage2_row("A", uncertainty=0.8), split, support_calibration(), permissive_policy())
    self.assertEqual(decision["selected_output"], "B")
    self.assertEqual([step["action"] for step in decision["transitions"]], ["OBSERVE", "CONTINUE", "REPLACE"])

def test_conflict_backtracks_then_later_branch_replaces(self):
    split = split_row(branches=[branch("B", "C"), branch("D", "D")])
    decision = replay_uncertainty_support(stage2_row("A"), split, support_calibration(), permissive_policy())
    self.assertIn("BACKTRACK", [step["action"] for step in decision["transitions"]])
    self.assertEqual(decision["selected_output"], "D")

def test_duplicate_render_or_bad_provenance_falls_back_exactly(self):
    original = stage2_row(["A", "B", "C", "D"])
    split = split_row(branches=[branch("B", "B", same_hash=True)], corrupt_rank=True)
    decision = replay_uncertainty_support(original, split, support_calibration(), permissive_policy())
    self.assertEqual(decision["selected_output"], original["output"])
    self.assertEqual(decision["selected_source"], "P0")
    self.assertEqual(decision["transitions"][-1]["action"], "FALLBACK_P0")
```

Also assert fixed six-branch visitation, first-view `CONTINUE`, exhausted `STOP_P0`, low optimistic upper-bound `BACKTRACK`, parse failures, raw-support-floor rejection, and atomic HR list projection.

- [ ] **Step 2: Run the new state tests and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_uncertainty_support -v`

Expected: failures for missing `UnifiedPolicy`, `candidate_snapshots`, and `replay_uncertainty_support`.

- [ ] **Step 3: Implement policy parsing and ordered snapshots**

```python
@dataclass(frozen=True)
class UnifiedPolicy:
    profile: str
    threshold: float
    raw_support_floor: float
    utility_calibrator: UtilityIsotonicCalibrator
    payload_sha256: str | None = None

@dataclass(frozen=True)
class CandidateSnapshot:
    branch_index: int
    revealed_roles: tuple[str, ...]
    output: Any
    canonical_answer: Hashable
    agreeing_hashes: tuple[str, ...]
    parseable_count: int
    features: AdvantageFeatures
    raw_score: float
    calibrated_advantage: float
    structurally_eligible: bool
```

Reuse `_split_audit`, `_validated_branches`, `_stage2_support`, `_p0_canonical_answer`, `_candidate_output`, and existing canonical projection helpers from `replay_split_search.py`. Do not copy dataset labels or benchmark-specific correctness into a snapshot or transition.

- [ ] **Step 4: Implement the single-score transition loop**

```python
for branch_index, branch in enumerate(branches):
    for role in ordered_roles(branch):
        reveal(role)
        snapshots = snapshots_for_revealed_views(...)
        if best.structurally_eligible and best.calibrated_advantage >= policy.threshold:
            return replace(best, action="REPLACE")
        if branch_can_reach_threshold(best, unrevealed_roles, policy):
            transition("CONTINUE", reason="utility_upper_bound_reachable")
            continue
        transition("BACKTRACK", reason="utility_upper_bound_below_threshold")
        break
return retain_p0(action="STOP_P0", reason="all_branches_exhausted")
```

The optimistic bound sets unrevealed agreement/support/gain/margin components to `1.0` while preserving the observed P0 uncertainty. It is evaluated by the same raw score and frozen utility calibrator; it is not a second threshold rule. Wrap schema/provenance/calibration/budget validation in one fail-closed boundary that returns the exact frozen P0 output and a `FALLBACK_P0` trace.

- [ ] **Step 5: Prove determinism and run split-search regressions**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_uncertainty_support tests.test_replay_split_search -v`

Expected: all tests pass; serializing two runs with canonical JSON produces identical bytes.

- [ ] **Step 6: Commit the state machine**

```bash
git add cvsearch/eval/replay_uncertainty_support.py tests/test_replay_uncertainty_support.py
git commit -m "feat: replay unified uncertainty support states"
```

### Task 3: Source-grouped OOF policy freeze

**Files:**
- Create: `cvsearch/eval/freeze_uncertainty_support.py`
- Create: `tests/test_freeze_uncertainty_support.py`

**Interfaces:**
- Consumes: development `Example` records extracted after label-blind snapshots, `PROFILES`, and `THRESHOLDS`.
- Produces: `source_group(benchmark: str, ordinal: int, input_image: str) -> str`, `select_configuration(examples: Sequence[Example]) -> Selection`, and `freeze_policy(examples: Sequence[Example], provenance: Mapping[str, Any]) -> dict[str, Any]`.

- [ ] **Step 1: Write failing source-group and leakage tests**

```python
class SourceGroupingTests(unittest.TestCase):
    def test_hr_resolutions_and_backbones_share_one_group(self):
        self.assertEqual(source_group("hr_bench_4k", 7, "4k.png"), source_group("hr_bench_8k", 7, "8k.png"))

    def test_non_hr_uses_normalized_source_path(self):
        self.assertEqual(source_group("vstar", 2, "./images/x.jpg"), "vstar:images/x.jpg")

    def test_oof_calibrator_never_sees_held_out_group(self):
        selection = select_configuration(grouped_examples())
        for fold in selection.folds:
            self.assertTrue(set(fold.train_groups).isdisjoint(fold.held_out_groups))
```

- [ ] **Step 2: Add failing selection, gate, and payload tests**

```python
def test_configuration_requires_no_harm_in_every_cell_and_pool(self):
    with self.assertRaises(NoFeasibleConfiguration):
        select_configuration(examples_with_one_negative_cell())

def test_tie_breaks_by_gain_corruptions_calls_then_declared_order(self):
    selected = select_configuration(tied_examples())
    self.assertEqual((selected.profile, selected.threshold), ("balanced", 0.05))

def test_frozen_payload_contains_no_labels_or_answers(self):
    payload = freeze_policy(grouped_examples(), provenance={"development_sha256": "a" * 64})
    serialized = json.dumps(payload, sort_keys=True)
    self.assertNotIn('"answer"', serialized)
    self.assertNotIn('"correct"', serialized)
    self.assertEqual(payload["payload_sha256"], canonical_payload_hash(payload))
```

- [ ] **Step 3: Run freeze tests and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_freeze_uncertainty_support -v`

Expected: `ModuleNotFoundError` for `cvsearch.eval.freeze_uncertainty_support`.

- [ ] **Step 4: Implement grouped examples and exact configuration ordering**

```python
@dataclass(frozen=True)
class Example:
    group: str
    backbone: str
    benchmark: str
    ordinal: int
    candidate_index: int
    features: AdvantageFeatures
    utility_target: float
    official_delta: int
    official_units: int
    observation_count: int

def utility_target(delta: int, official_units: int) -> float:
    if official_units <= 0 or abs(delta) > official_units:
        raise ValueError("official utility target is invalid")
    return 0.5 + delta / (2 * official_units)
```

For each held-out source group, fit the continuous calibrator on every other group, replay the held-out rows for each fixed profile/threshold, and aggregate cells, backbones, datasets, corrections, corruptions, and observations. Accept only configurations whose eight available cells and every pooled backbone/dataset have nonnegative official delta and whose corrections exceed corruptions. Sort feasible configurations by `(-net_gain, corruptions, observations, profile_index, threshold_index)`, then refit the selected profile on all development examples.

- [ ] **Step 5: Serialize a deterministic, label-free frozen policy**

```python
payload = {
    "schema_version": 1,
    "artifact_kind": "unified-uncertainty-support-policy",
    "data_scope": "opened_development_only",
    "profile": selection.profile,
    "weights": list(PROFILES[selection.profile]),
    "threshold": selection.threshold,
    "raw_support_floor": raw_support_floor,
    "utility_calibrator": selection.refit_calibrator.to_dict(),
    "development_inputs": dict(sorted(provenance.items())),
}
payload["payload_sha256"] = canonical_payload_hash(payload)
```

Only aggregate metrics and hashes belong in the policy. Per-row outputs, truth, utility targets, and correctness flags remain evaluator-local.

- [ ] **Step 6: Run focused and existing calibration tests**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_freeze_uncertainty_support tests.test_freeze_split_calibration tests.test_analyze_split_search -v`

Expected: all tests pass.

- [ ] **Step 7: Commit the development freezer**

```bash
git add cvsearch/eval/freeze_uncertainty_support.py tests/test_freeze_uncertainty_support.py
git commit -m "feat: freeze source grouped uncertainty policy"
```

### Task 4: Label separation, paired scoring, and gate reports

**Files:**
- Create: `cvsearch/eval/score_uncertainty_support.py`
- Create: `tests/test_score_uncertainty_support.py`

**Interfaces:**
- Consumes: aligned Stage-2/split JSONL files, frozen support calibrations, and the label-free unified policy.
- Produces: `generate_decisions(...) -> dict[str, Any]`, `score_decisions(...) -> dict[str, Any]`, `evaluate_locked_gates(report: Mapping[str, Any]) -> dict[str, Any]`, and a CLI with `freeze-development`, `generate-decisions`, and `score-decisions` subcommands.

- [ ] **Step 1: Write failing label-boundary and accounting tests**

```python
def test_decisions_do_not_change_when_label_fields_change(self):
    first = generate_decisions(strip_labels(stage2_rows), split_rows, calibration, policy)
    second = generate_decisions(replace_labels(stage2_rows), split_rows, calibration, policy)
    self.assertEqual(first, second)

def test_official_units_are_atomic_for_hr_and_scalar_elsewhere(self):
    report = score_decisions(labeled_rows(), frozen_decisions())
    self.assertEqual(report["cells"]["qwen/hr_bench_4k"]["official_units"], 48)
    self.assertEqual(report["aggregate"]["official_units"], 256)

def test_locked_gate_reports_each_exact_failure(self):
    gates = evaluate_locked_gates(report_with(qwen_hr4=(32, 48), total=(200, 256)))
    self.assertFalse(gates["passed"])
    self.assertIn("qwen/hr_bench_4k below 33/48", gates["failures"])
```

- [ ] **Step 2: Run score tests and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_score_uncertainty_support -v`

Expected: `ModuleNotFoundError` for `cvsearch.eval.score_uncertainty_support`.

- [ ] **Step 3: Implement strict loaders and decision generation**

```python
def generate_decisions(stage2_rows, split_rows, calibration, policy):
    aligned = align_by_ordinal(stage2_rows, split_rows)
    decisions = [
        label_free_decision_record(ordinal, replay_uncertainty_support(strip_evaluator_fields(stage2), split, calibration, policy))
        for ordinal, stage2, split in aligned
    ]
    return bind_canonical_hash({"schema_version": 1, "decisions": decisions})
```

Reject missing/duplicate ordinals, input SHA mismatch, policy hash mismatch, support-calibration mismatch, branch/rank/query provenance drift, unexpected topic/unit counts, and any decision containing `answer`, `correct`, `target`, or `candidate_correct` keys.

- [ ] **Step 4: Implement paired scoring and all locked gates**

```python
def evaluate_locked_gates(report):
    failures = []
    require_cell_no_harm(report, failures)
    require_dataset_no_harm(report, failures)
    require_backbone_no_harm(report, failures)
    if report["cells"]["qwen/hr_bench_4k"]["selected_correct"] < 33:
        failures.append("qwen/hr_bench_4k below 33/48")
    if report["aggregate"]["selected_correct"] <= 195:
        failures.append("aggregate did not exceed 195/256")
    if report["aggregate"]["corrections"] <= report["aggregate"]["corruptions"]:
        failures.append("corrections did not exceed corruptions")
    return {"passed": not failures, "failures": failures}
```

Reports include cells, pooled datasets, pooled backbones, aggregate counts, percentage-point changes, corrections, corruptions, selections, state/action counts, mean observations, oracle opportunity, Stage 3b `201/256` comparison, and exact input/policy/decision hashes. Keep per-row labels and answers out of committed reports.

- [ ] **Step 5: Expose deterministic CPU-only CLI commands**

Run contracts:

```bash
python -m cvsearch.eval.score_uncertainty_support freeze-development --split-root PATH --support-calibration PATH --policy-out PATH --report-out PATH
python -m cvsearch.eval.score_uncertainty_support generate-decisions --stage2-root PATH --split-root PATH --support-calibration PATH --policy PATH --decisions-out PATH
python -m cvsearch.eval.score_uncertainty_support score-decisions --stage2-root PATH --decisions PATH --report-out PATH --expected-topics 112 --expected-units 256
```

`freeze-development` may read labels only after snapshot generation. `generate-decisions` strips evaluator fields before replay. `score-decisions` is the only validation command allowed to read labels, and it receives the already-hashed decision artifact rather than a selector.

- [ ] **Step 6: Run focused scoring tests**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_score_uncertainty_support tests.test_analyze_split_search -v`

Expected: all tests pass and repeated CLI runs produce byte-identical JSON.

- [ ] **Step 7: Commit the scorer and separation boundary**

```bash
git add cvsearch/eval/score_uncertainty_support.py tests/test_score_uncertainty_support.py
git commit -m "feat: score locked uncertainty support replay"
```

### Task 5: Freeze on opened development and audit the selected policy

**Files:**
- Create: `reproduction/evidence_gap/adaptive_search_v8/frozen-policy-v1.json`
- Create: `reproduction/evidence_gap/reports/unified-uncertainty-support-development-v1.json`

**Interfaces:**
- Consumes: `adaptive_search_v7/raw/development/{backbone}/{benchmark}.jsonl` copied from the preserved Stage 3b worktree and `adaptive_search_v7/split-calibration-manifest-v2.json`.
- Produces: one immutable policy and one aggregate development report; validation data is absent from this command.

- [ ] **Step 1: Bind and stage the preserved development observations without modifying them**

Run:

```bash
mkdir -p reproduction/evidence_gap/adaptive_search_v8/raw/development
cp -a ../stage3-adaptive-split-search/reproduction/evidence_gap/adaptive_search_v7/raw/development/. reproduction/evidence_gap/adaptive_search_v8/raw/development/
sha256sum reproduction/evidence_gap/adaptive_search_v8/raw/development/*/*.jsonl
```

Expected: eight JSONL hashes match the Stage 3b source files recorded in the new development report; the raw directory remains ignored/uncommitted.

- [ ] **Step 2: Run source-grouped development freeze**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m cvsearch.eval.score_uncertainty_support freeze-development \
  --split-root reproduction/evidence_gap/adaptive_search_v8/raw/development \
  --support-calibration reproduction/evidence_gap/adaptive_search_v7/split-calibration-manifest-v2.json \
  --policy-out reproduction/evidence_gap/adaptive_search_v8/frozen-policy-v1.json \
  --report-out reproduction/evidence_gap/reports/unified-uncertainty-support-development-v1.json
```

Expected: a feasible configuration exists; every development OOF cell, pooled dataset, and pooled backbone is nonnegative; corrections exceed corruptions.

- [ ] **Step 3: Audit data scope, hashes, and determinism**

Run the same command with outputs in a `mktemp -d` directory, compare both outputs with `cmp`, and search both committed artifacts for `validation_v3`, per-row answers, or correctness arrays.

Expected: both `cmp` commands succeed; `data_scope` equals `opened_development_only`; no validation input/hash is present; the policy payload contains no labels or answers.

- [ ] **Step 4: Commit only the compact frozen artifacts**

```bash
git add reproduction/evidence_gap/adaptive_search_v8/frozen-policy-v1.json reproduction/evidence_gap/reports/unified-uncertainty-support-development-v1.json
git commit -m "exp: freeze unified uncertainty support policy"
```

### Task 6: Locked validation replay and exact regression gate

**Files:**
- Create: `reproduction/evidence_gap/adaptive_search_v8/locked-decisions-v1.json`
- Create: `reproduction/evidence_gap/reports/unified-uncertainty-support-locked-regression-v1.json`

**Interfaces:**
- Consumes: frozen policy, preserved validation Stage-2 and SPLIT observations, and support calibration.
- Produces: a label-free decision artifact first, followed by a separate scored locked-regression report.

- [ ] **Step 1: Stage immutable validation observations and verify their hashes**

Run:

```bash
mkdir -p reproduction/evidence_gap/adaptive_search_v8/raw/locked-regression
cp -a ../stage3-adaptive-split-search/reproduction/evidence_gap/adaptive_search_v7/raw/validation_v3/stage2 reproduction/evidence_gap/adaptive_search_v8/raw/locked-regression/
cp -a ../stage3-adaptive-split-search/reproduction/evidence_gap/adaptive_search_v7/raw/validation_v3/split reproduction/evidence_gap/adaptive_search_v8/raw/locked-regression/
sha256sum reproduction/evidence_gap/adaptive_search_v8/raw/locked-regression/{stage2,split}/*/*.jsonl
```

Expected: sixteen input files are present and their hashes are recorded in the decision artifact; raw files remain ignored/uncommitted.

- [ ] **Step 2: Generate frozen decisions before any validation scoring**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m cvsearch.eval.score_uncertainty_support generate-decisions \
  --stage2-root reproduction/evidence_gap/adaptive_search_v8/raw/locked-regression/stage2 \
  --split-root reproduction/evidence_gap/adaptive_search_v8/raw/locked-regression/split \
  --support-calibration reproduction/evidence_gap/adaptive_search_v7/split-calibration-manifest-v2.json \
  --policy reproduction/evidence_gap/adaptive_search_v8/frozen-policy-v1.json \
  --decisions-out reproduction/evidence_gap/adaptive_search_v8/locked-decisions-v1.json
```

Expected: 112 topic decisions; canonical decision hash is present; no truth/correctness fields are serialized.

- [ ] **Step 3: Score the immutable decisions exactly once**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m cvsearch.eval.score_uncertainty_support score-decisions \
  --stage2-root reproduction/evidence_gap/adaptive_search_v8/raw/locked-regression/stage2 \
  --decisions reproduction/evidence_gap/adaptive_search_v8/locked-decisions-v1.json \
  --report-out reproduction/evidence_gap/reports/unified-uncertainty-support-locked-regression-v1.json \
  --expected-topics 112 --expected-units 256
```

Expected gates: Qwen HR-Bench 4K is at least `33/48`; every one of eight cells, four datasets, and two backbones is no lower than CVSearch; aggregate is greater than `195/256`; corrections exceed corruptions; fixed geometry/ranking/observation/provenance/accounting audits pass.

- [ ] **Step 4: If the gate fails, diagnose without validation tuning**

Use the report only to identify the failing invariant or cell. Fix implementation defects with a new focused failing test. If behavior rather than implementation is responsible, return to opened-development OOF diagnostics and modify only a globally declared feature/state interpretation justified there; rerun Task 5 from scratch, produce a new policy hash, regenerate locked decisions, and never choose a threshold/profile from validation outcomes.

- [ ] **Step 5: Verify the requested tables and commit locked artifacts**

Confirm the report includes CVSearch, unified count/accuracy, net count and percentage-point change for HR-Bench 4K, HR-Bench 8K, TreeBench, V*, total, plus the Qwen/InternVL cell decomposition and a concise key conclusion.

```bash
git add reproduction/evidence_gap/adaptive_search_v8/locked-decisions-v1.json reproduction/evidence_gap/reports/unified-uncertainty-support-locked-regression-v1.json
git commit -m "exp: validate unified uncertainty support selector"
```

### Task 7: Full verification, review, and local main integration

**Files:**
- Modify only files found defective by verification or review.

**Interfaces:**
- Consumes: all implementation commits and compact artifacts from Tasks 1-6.
- Produces: a reviewed feature branch merged locally into `main` with the user's unrelated files untouched.

- [ ] **Step 1: Run the entire test suite from a clean feature worktree**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 2: Run artifact and repository audits**

```bash
git status --short
git diff main...HEAD --check
git diff --stat main...HEAD
rg -n 'validation_v3|candidate_correct|stage3_correct|correct_answer' reproduction/evidence_gap/adaptive_search_v8/frozen-policy-v1.json
```

Expected: only intentional feature files are tracked; diff check is clean; the policy scan returns no matches; raw observation directories are not tracked.

- [ ] **Step 3: Request a requirements and code-quality review**

Review against `docs/superpowers/specs/2026-08-13-unified-uncertainty-support-selector-design.md`, paying special attention to label leakage, exact P0 fallback, single-score transitions, grouped folds, deterministic hashes, locked gates, and preservation of existing observations. Convert each valid finding into a focused failing test before changing code.

- [ ] **Step 4: Re-run targeted and full verification after review fixes**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_uncertainty_support tests.test_freeze_uncertainty_support tests.test_score_uncertainty_support -v
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v
git diff main...HEAD --check
```

Expected: all commands pass and the locked report still has `gates.passed == true`.

- [ ] **Step 5: Merge the verified branch locally**

From the main checkout, record the existing untracked files, run `git merge --no-ff feature/unified-uncertainty-support-selector`, rerun the focused tests on `main`, and compare the untracked-file list before/after.

Expected: merge succeeds without overwriting user work; focused tests pass on `main`; unrelated untracked files remain unchanged.

- [ ] **Step 6: Report evidence and positive progress**

Report the merge commit, test counts, selected profile/threshold, development OOF change, locked per-dataset and per-backbone table, corrections/corruptions, observation cost, Stage 3b comparison, and which previously corrupted cells were recovered.
