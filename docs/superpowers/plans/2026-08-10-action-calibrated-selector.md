# Action-Calibrated Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one label-blind selector that admits EXPAND at gain `>= 0.25`, admits ZOOM only at gain `> 0.50`, and produces verified full scores `171/191`, `616/800`, and `618/800` from the existing phase-6 observations.

**Architecture:** Represent each action boundary as an immutable frozen rule in phase 5, bind the ordered rule material into the phase-6 decision digest, and make the final scorer validate and publish that same material without reopening selection. Preserve the v1 bundles and publish v2 bundles to a new commit-bound directory.

**Tech Stack:** Python 3.11, standard-library `dataclasses`/`unittest`, existing phase-6 selector and scorer, canonical SHA-256 provenance.

## Global Constraints

- One selector and identical ordered admission rules apply to V*, HR-Bench 4K, and HR-Bench 8K.
- EXPAND uses `stability_gain >= 0.25`; ZOOM uses `stability_gain > 0.50`.
- Tie order remains `EXPAND`, then `ZOOM`; no epsilon or rounding is allowed.
- Benchmark, resolution, question type, category, ordinal, labels, and correctness remain forbidden selector inputs.
- Existing raw Qwen/SAM observations and `full_b080f86_fda2ccd` bundles are immutable.
- Trusted annotations are opened only after complete label-blind decision freeze.
- V* must remain at least `171/191`; both HR resolutions must strictly exceed their execution-local paired P0.

---

### Task 1: Encode and test action-specific admission

**Files:**

- Modify: `tests/test_phase5_unified_selector.py`
- Modify: `cvsearch/eval/phase5_unified_selector.py`

**Interfaces:**

- Produces: frozen `ActionAdmissionRule(action: str, operator: str, threshold: float)`.
- Produces: `ACTION_ADMISSION_RULES: tuple[ActionAdmissionRule, ...]` in canonical EXPAND/ZOOM order.
- Preserves: `select_unified_state(p0, candidates) -> SelectionDecision`.

- [ ] **Step 1: Write failing boundary tests**

Replace the old shared-threshold test with exact per-action tests:

```python
def test_expand_exact_quarter_gain_is_admitted(self):
    decision = select_unified_state(
        self.p0(0.5), [self.candidate("EXPAND", 0.75)],
    )
    self.assertEqual(decision.action, "EXPAND")
    self.assertEqual(decision.stability_gain, 0.25)

def test_zoom_exact_half_gain_is_rejected(self):
    decision = select_unified_state(
        self.p0(0.25), [self.candidate("ZOOM", 0.75)],
    )
    self.assertEqual(decision.action, "P0")
    self.assertIsNone(decision.stability_gain)

def test_zoom_gain_above_half_is_admitted(self):
    decision = select_unified_state(
        self.p0(0.25),
        [self.candidate("ZOOM", math.nextafter(0.75, math.inf))],
    )
    self.assertEqual(decision.action, "ZOOM")
    self.assertGreater(decision.stability_gain, 0.5)
```

Update the below-threshold test to use EXPAND and update the tie test so both
gains are above `0.5`; this ensures it still exercises a real admitted tie.

- [ ] **Step 2: Run the phase-5 test and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_phase5_unified_selector -v
```

Expected: the exact-half ZOOM test fails because the v1 selector admits it.

- [ ] **Step 3: Implement the minimal immutable rule contract**

Add before `SelectionDecision`:

```python
@dataclass(frozen=True)
class ActionAdmissionRule:
    action: str
    operator: str
    threshold: float


ACTION_ADMISSION_RULES = (
    ActionAdmissionRule("EXPAND", ">=", 0.25),
    ActionAdmissionRule("ZOOM", ">", 0.50),
)
```

Add a closed comparison helper:

```python
def _is_admitted(action: str, gain: float) -> bool:
    rule = next(
        (item for item in ACTION_ADMISSION_RULES if item.action == action),
        None,
    )
    if rule is None:
        raise RuntimeError("candidate action has no admission rule")
    if rule.operator == ">=":
        return gain >= rule.threshold
    if rule.operator == ">":
        return gain > rule.threshold
    raise RuntimeError("candidate action admission operator is unsupported")
```

Replace the shared threshold filter with:

```python
admitted = [item for item in validated if _is_admitted(item[1]["action"], item[0])]
```

- [ ] **Step 4: Run phase-5 GREEN**

Run the Task 1 command again. Expected: all phase-5 selector tests pass.

- [ ] **Step 5: Commit the selector behavior**

```bash
git add cvsearch/eval/phase5_unified_selector.py tests/test_phase5_unified_selector.py
git commit -m "feat: calibrate selector admission by action"
```

---

### Task 2: Bind ordered admission rules into frozen provenance

**Files:**

- Modify: `tests/test_phase6_combined_selection.py`
- Modify: `cvsearch/eval/phase6_combined_selection.py`

**Interfaces:**

- Consumes: `phase5.ActionAdmissionRule` and `phase5.ACTION_ADMISSION_RULES`.
- Produces: `REVIEWED_ADMISSION_RULES` and selector ID `phase5-action-stability-ege025-zgt050-v2`.
- Produces: `FrozenCombinedDecisionBatch.admission_rules` in place of the scalar `threshold`.

- [ ] **Step 1: Write failing provenance tests**

Change the exact phase-5 binding assertion to:

```python
self.assertEqual(
    phase6.REVIEWED_ADMISSION_RULES,
    (
        phase5.ActionAdmissionRule("EXPAND", ">=", 0.25),
        phase5.ActionAdmissionRule("ZOOM", ">", 0.5),
    ),
)
self.assertEqual(frozen.admission_rules, phase6.REVIEWED_ADMISSION_RULES)
```

Add a mutation test that replaces `admission_rules` with a rule using `>=` for
ZOOM, verifies `recompute_digest()` changes, and verifies `verify_digest()`
rejects the mutated batch.

- [ ] **Step 2: Run phase-6 selection tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_phase6_combined_selection -v
```

Expected: failures for missing `REVIEWED_ADMISSION_RULES` and missing batch
`admission_rules`.

- [ ] **Step 3: Replace scalar threshold provenance**

Import the rule type and constant from phase 5, define:

```python
SELECTOR_ID = "phase5-action-stability-ege025-zgt050-v2"
REVIEWED_ADMISSION_RULES = (
    phase5.ActionAdmissionRule("EXPAND", ">=", 0.25),
    phase5.ActionAdmissionRule("ZOOM", ">", 0.50),
)
```

Change `FrozenCombinedDecisionBatch` to:

```python
@dataclass(frozen=True)
class FrozenCombinedDecisionBatch(Sequence[FrozenCombinedDecision]):
    records: tuple[FrozenCombinedDecision, ...]
    selector_id: str
    selector_source_sha256: str
    admission_rules: tuple[phase5.ActionAdmissionRule, ...]
    tie_order: tuple[str, str]
    canonical_digest: str
```

Serialize the rules in `digest_material()` with:

```python
"admission_rules": [asdict(rule) for rule in self.admission_rules],
```

At freeze time require exact equality between
`phase5.ACTION_ADMISSION_RULES` and `REVIEWED_ADMISSION_RULES`. Use the same
ordered `asdict` list in digest material and store the reviewed tuple on the
batch. Remove every scalar `threshold` field.

- [ ] **Step 4: Bind the selector source hash**

Run:

```bash
sha256sum cvsearch/eval/phase5_unified_selector.py
```

Set `SELECTOR_SOURCE_SHA256` to the exact 64-hex digest printed by this command.

- [ ] **Step 5: Run phase-6 selection GREEN**

Run the Task 2 test command. Expected: every combined-selection test passes.

- [ ] **Step 6: Commit provenance binding**

```bash
git add cvsearch/eval/phase6_combined_selection.py tests/test_phase6_combined_selection.py
git commit -m "fix: bind action admission rules into selection"
```

---

### Task 3: Make the final scorer enforce and publish v2 rules

**Files:**

- Modify: `tests/test_phase6_final_scorer.py`
- Modify: `cvsearch/eval/phase6_final_scorer.py`

**Interfaces:**

- Consumes: `FrozenCombinedDecisionBatch.admission_rules` and
  `phase6.REVIEWED_ADMISSION_RULES`.
- Produces: manifest `decision_freeze.admission_rules`.
- Preserves: scorer label boundary, atomic bundle publication, and suite-level
  shared-selector digest.

- [ ] **Step 1: Write failing scorer/provenance tests**

Update expected selected rows to the v2 selector ID. Assert the manifest has
exactly:

```python
[
    {"action": "EXPAND", "operator": ">=", "threshold": 0.25},
    {"action": "ZOOM", "operator": ">", "threshold": 0.5},
]
```

Add subtests that replace a prepared batch's rules with a missing rule, reversed
order, ZOOM `>=`, or ZOOM threshold `0.49`; `_selected_material` must reject
each before trusted annotation scoring.

- [ ] **Step 2: Run scorer tests and verify RED**

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_phase6_final_scorer -v
```

Expected: failures because the scorer still expects scalar `threshold` and the
v1 selector ID.

- [ ] **Step 3: Validate action-specific candidate boundaries**

In `_selected_material`, replace the scalar identity check with exact rule
tuple equality. For a selected action, find its rule in the frozen tuple and
validate the gain with the explicit operator. Reject an unknown action or
operator. Do not import or rerun `select_unified_state`.

Replace manifest material:

```python
"admission_rules": [
    asdict(rule) for rule in prepared.frozen_batch.admission_rules
],
```

Replace the suite shared-selector key `threshold` with `admission_rules`.

- [ ] **Step 4: Run scorer GREEN and the focused phase suite**

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_phase5_unified_selector tests.test_phase6_combined_selection tests.test_phase6_final_scorer -v
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit scorer enforcement**

```bash
git add cvsearch/eval/phase6_final_scorer.py tests/test_phase6_final_scorer.py
git commit -m "fix: enforce action rules in final scoring"
```

---

### Task 4: Replay, publish, audit, and report all three benchmarks

**Files:**

- Create through atomic scorer: `reproduction/evidence_gap/phase6_selected/full_b080f86_<code-revision>/vstar/`
- Create through atomic scorer: `reproduction/evidence_gap/phase6_selected/full_b080f86_<code-revision>/hr-bench_4k/`
- Create through atomic scorer: `reproduction/evidence_gap/phase6_selected/full_b080f86_<code-revision>/hr-bench_8k/`
- Modify with verified values: `reproduction/evidence_gap/reports/phase6-full-result-snapshot.json`

**Interfaces:**

- Consumes: existing disabled/enabled raw paths under
  `reproduction/evidence_gap/phase6_full/b080f86/<benchmark>/`.
- Produces: three atomic selected bundles and one suite report.

- [ ] **Step 1: Run the complete CPU suite**

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures or errors.

- [ ] **Step 2: Perform one scorer-only full replay**

Call `score_phase6_suite` once with these exact pairs for each benchmark:

```text
disabled: reproduction/evidence_gap/phase6_full/b080f86/<benchmark>/disabled.jsonl
combined: reproduction/evidence_gap/phase6_full/b080f86/<benchmark>/enabled.jsonl
profile: full
```

The output root must include the short implementation commit SHA and must not
already exist. Print the returned suite report to stdout for review. Expected
scores are V* `171/191`, HR-4K `616/800`, HR-8K `618/800`, with
`all_three_full_success=true`.

- [ ] **Step 3: Audit immutable artifacts**

For every benchmark, compare `sha256sum selected.jsonl` to
`.selected_jsonl.sha256` in its manifest. Verify every manifest contains the
exact ordered `admission_rules`, the v2 selector ID, current selector source
hash, and `completed_before_trusted_label_read=true`. Verify no
`perform_EGSearch` process remains.

- [ ] **Step 4: Update the compact snapshot using apply_patch**

Preserve the v1 result block, add a v2 published-result block with exact bundle
paths, scores, action counts, source/manifest/selected hashes, and mark the
read-only replay as realized. Keep the evaluation classification
`aggregate-exposed internal engineering evaluation`.

- [ ] **Step 5: Commit published artifacts and report**

Stage only the three new selected JSONL files, three manifests, the compact
snapshot, and any suite report created deliberately. Do not stage raw phase-6
JSONLs, lock files, caches, or test-generated directories.

```bash
git commit -m "data: publish action-calibrated phase6 bundles"
```

- [ ] **Step 6: Final verification**

Repeat the complete CPU suite, exact bundle hash audit, `git diff --check`, and
remote-free local status inspection. Report the actual test count and exact
scores; do not claim untouched-test evidence.

## Self-Review

- Spec coverage: action boundaries, tie order, label blindness, provenance,
  scorer validation, immutable v1 bundles, v2 publication, and evaluation
  integrity each map to a task above.
- Placeholder scan: no TODO/TBD or unspecified implementation step remains;
  the only computed value is the selector SHA-256, obtained by an exact command
  after the source is finalized.
- Type consistency: `ActionAdmissionRule` and `admission_rules` are defined in
  Task 1, bound in Task 2, and consumed unchanged in Tasks 3 and 4.
