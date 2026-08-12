# Calibrated Adaptive-Scale Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-neutral, calibrated action controller after the frozen Phase 1 ranking and validate Qwen ZOOM/EXPAND candidates without changing Phase 1 order or answers on failure.

**Architecture:** A new pure controller converts sanitized evidence requirements and normalized support observations into bounded action decisions. Existing Qwen observation producers remain candidate-only and are made compatible with the exact adaptive-ranking configuration; a separate evaluator fits calibration, replays decisions, and reports answer, calibration, action, trajectory, cost, and Phase 1 preservation metrics.

**Tech Stack:** Python 3.11 standard library, existing `unittest` suite, existing CVSearch/Qwen/SAM/CLIP runtime, JSON/JSONL reports.

## Global Constraints

- Commit `c1ee03a` remains the immutable Phase 1 ranking baseline.
- Do not modify `cvsearch/evidence_gap/query_profile.py`, `cvsearch/evidence_gap/ranking.py`, or `reproduction/evidence_gap/configs/dev_adaptive_ranking_v1.json`.
- The controller must reject benchmark, resolution, category, ordinal, label, ground-truth, target-box, and correctness fields.
- Disabled or failed Phase 2 must retain the exact Phase 1 output and rank trace.
- No benchmark-specific action thresholds or routes.
- SPLIT remains a controller action but no new split candidate generator is admitted until ZOOM/EXPAND candidate diagnostics demonstrate unresolved localization ambiguity.

---

### Task 1: Pure evidence demand, calibration, and action controller

**Files:**
- Create: `cvsearch/evidence_gap/adaptive_controller.py`
- Test: `tests/test_evidence_gap_adaptive_controller.py`

**Interfaces:**
- Consumes: sanitized query-plan dictionaries, normalized support observations, available action names, support trajectory, action costs.
- Produces: `EvidenceDemand`, `SupportObservation`, `IsotonicCalibrator`, `ActionDecision`, `infer_evidence_demand()`, `fit_isotonic()`, and `select_adaptive_action()`.

- [ ] **Step 1: Write failing demand and input-boundary tests**

```python
def test_relation_and_detail_create_mixed_soft_prior(self):
    demand = infer_evidence_demand((
        {"kind": "target_detail", "target": "number", "requirements": ["presence", "visual_detail"]},
        {"kind": "relation_context", "targets": ["entrance", "woman"]},
    ))
    self.assertGreater(demand.detail, 0.0)
    self.assertGreater(demand.context, 0.0)
    self.assertAlmostEqual(sum(demand.action_prior().values()), 1.0)

def test_policy_input_rejects_evaluator_metadata(self):
    with self.assertRaises(ValueError):
        infer_evidence_demand(({"kind": "target_detail", "label": 0},))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_adaptive_controller -v`

Expected: import failure for missing `adaptive_controller`.

- [ ] **Step 3: Implement immutable DTOs and evidence-demand inference**

```python
@dataclass(frozen=True)
class EvidenceDemand:
    detail: float
    context: float
    localization: float

    def action_prior(self) -> dict[str, float]:
        weights = {ZOOM: self.detail, EXPAND: self.context, SPLIT: self.localization}
        total = sum(weights.values())
        return ({name: value / total for name, value in weights.items()}
                if total else {ZOOM: 1 / 3, EXPAND: 1 / 3, SPLIT: 1 / 3})
```

Validate exact JSON builtins and reject forbidden metadata before reading any values.

- [ ] **Step 4: Write failing calibration tests**

```python
def test_isotonic_fit_is_monotone_and_bounded(self):
    calibrator = fit_isotonic(((0.1, 0), (0.2, 1), (0.3, 0), (0.9, 1)))
    values = [calibrator.predict(value) for value in (0.0, 0.2, 0.3, 1.0)]
    self.assertEqual(values, sorted(values))
    self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
```

- [ ] **Step 5: Run the calibration test and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_adaptive_controller.AdaptiveCalibrationTest -v`

Expected: failure because `fit_isotonic` is missing.

- [ ] **Step 6: Implement standard-library PAVA isotonic calibration**

Implement stable sorting, adjacent-violator pooling, endpoint prediction, strict finite validation, and immutable knot/value tuples. No external dependency is added.

- [ ] **Step 7: Write failing action-policy tests**

```python
def test_observed_missing_reason_overrides_soft_prior(self):
    decision = select_adaptive_action(
        EvidenceDemand(detail=1.0, context=0.0, localization=0.0),
        SupportObservation(0.1, 0.8, 0.1, 1.0, 1.0, "context_missing", 0.0),
        available=(ZOOM, EXPAND), trajectory=(0.2,), has_unvisited_branch=True,
    )
    self.assertEqual(decision.action, EXPAND)

def test_two_stalled_gains_backtrack_without_deleting_patch(self):
    decision = select_adaptive_action(
        EvidenceDemand(0.3, 0.3, 0.4), observation(),
        available=(ZOOM, EXPAND, BACKTRACK), trajectory=(0.40, 0.41, 0.40),
        has_unvisited_branch=True,
    )
    self.assertEqual(decision.action, BACKTRACK)
```

- [ ] **Step 8: Run the policy tests and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_adaptive_controller.AdaptiveActionPolicyTest -v`

Expected: failure because `select_adaptive_action` is missing.

- [ ] **Step 9: Implement the minimal decaying-prior policy**

Use missing-reason routing first, two-step stall backtracking second, calibrated sufficient-and-stable stopping third, and a bounded prior/cost score only for remaining ties. Return an append-only decision record with no mutation or deletion methods.

- [ ] **Step 10: Run the full controller test file and commit**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_adaptive_controller -v`

Expected: all controller tests pass.

Commit: `git commit -m "feat: add calibrated adaptive search controller"`

---

### Task 2: Cross-dataset calibration and search metrics

**Files:**
- Create: `cvsearch/eval/eval_adaptive_search.py`
- Test: `tests/test_eval_adaptive_search.py`

**Interfaces:**
- Consumes: exact row-level records containing reference outcome, selected outcome, raw/calibrated support, action, stop decision, cost, and rank-trace digest.
- Produces: `evaluate_adaptive_rows(rows)` with accuracy, corrections/corruptions, false stops, ECE, Brier, action gain, backtrack recovery, cost, and rank-preservation fields.

- [ ] **Step 1: Write failing metric tests with hand-computable rows**

```python
def test_reports_accuracy_calibration_false_stop_cost_and_rank_identity(self):
    report = evaluate_adaptive_rows(self.rows())
    self.assertEqual(report["paired"]["corrections"], 1)
    self.assertEqual(report["paired"]["corruptions"], 0)
    self.assertEqual(report["safety"]["false_stops"], 1)
    self.assertAlmostEqual(report["calibration"]["brier"], 0.065)
    self.assertTrue(report["phase1"]["all_rank_traces_preserved"])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_eval_adaptive_search -v`

Expected: import failure for missing evaluator.

- [ ] **Step 3: Implement strict metrics with fixed ten-bin ECE**

Use topic rows as the unit, exact semantic correctness booleans supplied only at evaluator time, action-stratified counts, mean added calls/pixels/latency, and SHA-256 rank-trace equality. Reject duplicate identities and nonfinite values.

- [ ] **Step 4: Add leakage and malformed-input tests**

Test duplicate row IDs, invalid probabilities, missing rank digests, nonfinite costs, and policy records containing evaluator fields.

- [ ] **Step 5: Run tests and commit**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_eval_adaptive_search -v`

Expected: all evaluator tests pass.

Commit: `git commit -m "feat: evaluate adaptive search calibration and cost"`

---

### Task 3: Attach existing ZOOM/EXPAND observations to frozen Phase 1

**Files:**
- Modify: `cvsearch/evidence_gap/method.py`
- Create: `reproduction/evidence_gap/configs/dev_adaptive_ranking_observe_v2.json`
- Modify: `tests/test_evidence_gap_method.py`
- Test: `tests/test_evidence_gap_adaptive_observation.py`

**Interfaces:**
- Consumes: exact Phase 1 adaptive ranking config plus existing disabled NEXT, enabled P2C ZOOM, and enabled P4A EXPAND observation groups.
- Produces: a candidate-only trace with unchanged Phase 1 output and the exact original `candidate_ranks` prefix.

- [ ] **Step 1: Write a failing config-composition test**

```python
def test_frozen_adaptive_rank_accepts_candidate_only_zoom_expand(self):
    config = load_method_config("reproduction/evidence_gap/configs/dev_adaptive_ranking_observe_v2.json")
    self.assertEqual(config["beta"], 1.0)
    self.assertEqual(config["detail_alpha_discount"], 0.15)
    self.assertTrue(config["p2c_zoom_enabled"])
    self.assertTrue(config["p4a_expand_enabled"])
    self.assertFalse(config["p2c_zoom_replacement_enabled"])
    self.assertFalse(config["p4a_expand_replacement_enabled"])
```

- [ ] **Step 2: Run the config test and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_method -v`

Expected: failure because the composed config is absent or rejected.

- [ ] **Step 3: Generalize only the observation compatibility guard**

Keep all legacy frozen-config checks. Add a second accepted profile whose ranking fields exactly equal Phase 1 (`query_linear`, `1.0/0.25/1.0`, detail discount `0.15`, context gain `0.45`, quick gate `0.8`) and whose action replacement flags are false. Do not modify ranking implementation or weights.

- [ ] **Step 4: Write an end-to-end fake-runtime preservation test**

Run the same synthetic search once with Phase 2 disabled and once with candidate observation enabled. Assert identical emitted output, identical candidate identity/order trace, action audits present only in the enabled run, and fail-closed behavior for model, budget, and render errors.

- [ ] **Step 5: Run focused method/observation tests and commit**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_method tests.test_evidence_gap_adaptive_observation -v`

Expected: all focused tests pass.

Commit: `git commit -m "feat: observe scale actions after frozen ranking"`

---

### Task 4: Label-blind replay selector for action candidates

**Files:**
- Create: `cvsearch/eval/replay_adaptive_search.py`
- Test: `tests/test_replay_adaptive_search.py`

**Interfaces:**
- Consumes: Phase 1 row, candidate-only observation row, a frozen isotonic calibrator, and the pure adaptive controller.
- Produces: selected output plus an exact decision audit; trusted labels are accepted only by the separate scoring function after decisions are frozen.

- [ ] **Step 1: Write failing extraction and selection tests**

Test mixed detail/context demand, semantic HR option normalization, action support gains, P0 retention on missing calibration, strong evidence overriding prior, and exact Phase 1 fallback when no action is admitted.

- [ ] **Step 2: Run the replay tests and verify RED**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_adaptive_search -v`

Expected: import failure for missing replay module.

- [ ] **Step 3: Implement strict candidate extraction and decision replay**

Reuse existing `ZoomAudit`, `ExpandAudit`, `aggregate_hr_answers`, and `aggregate_vstar_losses` contracts. The selector sees support probabilities, candidate stability, demand, cost, and action availability but never correctness or benchmark metadata.

- [ ] **Step 4: Add a frozen calibration-selection routine**

Fit only on declared calibration rows, serialize calibrator knots and hashes, freeze decisions, then open evaluator labels for scoring. Require one shared decision rule across V*, HR-4K, and HR-8K.

- [ ] **Step 5: Run replay tests and commit**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_replay_adaptive_search -v`

Expected: all replay tests pass.

Commit: `git commit -m "feat: replay calibrated scale-action decisions"`

---

### Task 5: Qwen development experiment and failure-directed correction

**Files:**
- Create: `reproduction/evidence_gap/adaptive_search_v2/`
- Create: `reproduction/evidence_gap/reports/adaptive-search-v2-dev.json`
- Modify only if a predeclared test fails: controller, evaluator, observation compatibility, or replay files from Tasks 1-4.

**Interfaces:**
- Consumes: frozen Phase 1 30-topic V* partition, source-grouped HR development partitions, exact model/SAM/CLIP artifacts, and the candidate-only config.
- Produces: paired candidate observations, frozen calibration/decision manifest, and a metrics report.

- [ ] **Step 1: Re-run frozen Phase 1 and verify its digest/metrics**

Run the checked-in adaptive config on the exact 30 V* ordinals and require the checked-in answer/rank report to match before generating actions.

- [ ] **Step 2: Generate candidate-only ZOOM/EXPAND observations**

Run V*, HR-4K, and HR-8K development partitions on available H200 GPUs with exact launch manifests and no replacement enabled.

- [ ] **Step 3: Fit calibration and freeze replay decisions**

Keep HR 4K/8K versions of each topic grouped. Fit on the calibration partition only, serialize hashes, and freeze selected actions before scoring.

- [ ] **Step 4: Score all required metrics**

Report paired accuracy, corrections/corruptions, false stops, ECE/Brier, action gains, backtrack diagnostics, calls/pixels/latency, and Phase 1 rank identity.

- [ ] **Step 5: Perform failure-directed correction without touching Phase 1**

If the gate fails, attribute failure to candidate generation, calibration, selection, answer production, or cost. Change one Stage 2 factor, add a failing regression test, rerun the affected development partition, and retain the simpler version unless the new factor improves the minimum dataset delta without rank drift.

- [ ] **Step 6: Commit the frozen development result**

Commit code, configs, compact reports, and manifests. Do not commit model files, source images, or large raw JSONL artifacts.

---

### Task 6: Cross-model and extra-dataset validation gate

**Files:**
- Modify: `cvsearch/perform_EGSearch.py`
- Modify: `cvsearch/models/modeling_internvl.py` or `cvsearch/models/modeling_llava.py`
- Test: `tests/test_evidence_gap_model_adapter.py`
- Create: `reproduction/evidence_gap/reports/adaptive-search-v2-transfer.json`

**Interfaces:**
- Consumes: the same normalized controller contract and one non-Qwen backbone already supported by `perform_CVSearch.py`.
- Produces: model-specific calibrated support values with shared controller decisions and transfer metrics on HR plus TreeBench or MME-RealWorld-Lite.

- [ ] **Step 1: Write a failing adapter conformance test**

Require each adapter to expose answer production and normalized evidence support without leaking model-specific logits into the controller DTO.

- [ ] **Step 2: Generalize runtime dispatch by model config type**

Reuse the repository's Qwen/InternVL/LLaVA dispatch. Keep rendering and raw confidence inside the adapter and pass only normalized support/answer records to the controller.

- [ ] **Step 3: Run one second backbone and one additional benchmark**

Use the same action rule and stop threshold. Fit a backbone-specific calibrator on the declared calibration partition, then evaluate the held-out partition.

- [ ] **Step 4: Apply the transfer gate**

Require lower calibrated ECE/Brier, zero rank drift, no dataset/model point-estimate regression, and positive macro paired delta before claiming cross-model or cross-dataset success.

- [ ] **Step 5: Commit transfer report or documented blocker**

If checkpoint/runtime incompatibility blocks execution, retain all conformance tests and report the exact external blocker without weakening the Qwen or dataset gates.

---

### Task 7: Final regression and integrity verification

**Files:**
- Verify only; modify files only through a new failing regression test.

**Interfaces:**
- Consumes: all committed Phase 2 code and compact reports.
- Produces: final test evidence, clean diff, and Phase 1 integrity hashes.

- [ ] **Step 1: Run focused Phase 1 preservation tests**

Run ranking, query-profile, replay-ranking, and V* Recall evaluator tests.

- [ ] **Step 2: Run the complete unit suite**

Run: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -q`

Expected: 575 existing tests plus new tests pass, with only artifact-availability skips.

- [ ] **Step 3: Verify immutable Phase 1 file hashes**

Compare `query_profile.py`, `ranking.py`, the adaptive config, and the checked-in ranking report against commit `c1ee03a`.

- [ ] **Step 4: Inspect final diff and repository status**

Run `git diff --check`, review every changed path, and keep temporary artifact symlinks and large inference outputs uncommitted.

- [ ] **Step 5: Commit final verification metadata**

Commit only compact, reproducible verification metadata if it adds information beyond the experiment report.
