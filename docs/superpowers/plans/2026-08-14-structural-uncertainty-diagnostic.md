# Structural Uncertainty Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether prefix-visible candidate persistence or conservative leave-one-view-out stability adds enough transferable signal to improve the honest nested current-pool result without new inference.

**Architecture:** Run an isolated CPU diagnostic over the five already opened partitions. Candidate-history features are computed only from checkpoints at or before the current candidate; a shared grouped-OOF benefit/harm head and one shared action grid are selected without dataset/backbone routing. Production code is changed only if the current two-partition outer result improves while every safety and cost gate remains satisfied.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, existing CVSearch risk-topic replay.

## Global Constraints

- Use `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11`.
- Use only the five opened development partitions; do not access MME-RealWorld-Lite.
- Do not use dataset, backbone, ordinal, category, evaluator answer, correctness, or GT geometry as inference features.
- Preserve exact P0 fallback and the frozen Stage-1/2/3 evidence trajectory.
- Use outer partition isolation and inner source-group isolation.
- Require both backbones strictly positive, all eight cells nonnegative, corrections above corruptions, and mean observations at most 12.8 before production implementation.

---

### Task 1: Run the prefix-history diagnostic

**Files:**
- Create: `/tmp/cvsearch_structural_stability_proto.py`
- Read: `/tmp/robust_verifier_proto.py`
- Read: `cvsearch/eval/freeze_uncertainty_support.py`

**Interfaces:**
- Consumes: `_risk_topics(records) -> tuple[_RiskTopic, ...]`.
- Produces: one JSON line per outer partition plus combined and current-pool metrics.

- [ ] **Step 1: Define the prefix-only structural vector**

```python
def structural_features(topic, index):
    current = topic.examples[index]
    prefix = topic.examples[:index + 1]
    history = [item for item in prefix
               if item.candidate_canonical == current.candidate_canonical]
    checkpoints = tuple(dict.fromkeys(item.checkpoint for item in prefix))
    present = {item.checkpoint for item in history}
    consecutive = 0
    for checkpoint in reversed(checkpoints):
        if checkpoint not in present:
            break
        consecutive += 1
    strongest = round(
        current.agreeing_views / current.features.agreement
        - current.agreeing_views
    )
    return np.asarray((
        len(history) / len(checkpoints),
        (current.observations - history[0].observations) / 14.0,
        consecutive / len(checkpoints),
        len({item.checkpoint[0] for item in history}) / 6.0,
        min(item.evidence_features.mean_support for item in history),
        max(item.evidence_features.mean_support for item in history),
        max(item.evidence_features.mean_support for item in history)
        - min(item.evidence_features.mean_support for item in history),
        min(item.features.agreement for item in history),
        max(item.features.agreement for item in history),
        (current.agreeing_views - history[0].agreeing_views + 14) / 28.0,
        float(current.agreeing_views >= 3),
        (current.agreeing_views - strongest + 14) / 28.0,
    ))
```

- [ ] **Step 2: Assert prefix integrity and deterministic bounds**

```python
first = structural_features(topic, 0)
assert first.shape == (12,)
assert np.all((0.0 <= first) & (first <= 1.0))
assert np.array_equal(first, structural_features(topic, 0))
```

- [ ] **Step 3: Run five-direction outer replay**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 \
  /tmp/cvsearch_structural_stability_proto.py
```

Expected: deterministic JSON metrics for `development`, `final_v2`,
`validation_v1`, `validation_v2`, and `validation_v3`, followed by combined and
current-pool acceptance failures.

- [ ] **Step 4: Apply the predeclared promotion gate**

Promote Task 2 only if current-pool outer net exceeds `+4/512`, both current
backbones are strictly positive, all eight current cells are nonnegative,
corrections exceed corruptions, and mean observations are at most 12.8. If the
gate fails, write an authenticated negative report and stop this family.

### Task 2: Measure conservative deletion stability

**Files:**
- Modify: `/tmp/cvsearch_structural_stability_proto.py`
- Read: `cvsearch/eval/replay_uncertainty_support.py`

**Interfaces:**
- Consumes: frozen `_PreparedReplay` views at each candidate checkpoint.
- Produces: `deletion_survival_fraction`, `worst_deleted_agreement`, and `strict_vote_dominance_after_every_deletion` as label-blind features.

- [ ] **Step 1: Recompute each checkpoint after deleting one observed view**

```python
def deletion_stability(observed, p0, candidate):
    outcomes = []
    for removed in range(len(observed)):
        kept = observed[:removed] + observed[removed + 1:]
        counts = Counter(view.canonical_answer for view in kept
                         if view.canonical_answer is not None)
        outcomes.append(counts.get(candidate, 0) >= 2
                        and counts.get(candidate, 0)
                        > max((count for answer, count in counts.items()
                               if answer != candidate), default=0))
    return sum(outcomes) / len(outcomes), all(outcomes)
```

- [ ] **Step 2: Repeat Task 1 outer replay with only the three deletion features added**

Run the Task 1 command. Expected: a second deterministic result block labelled
`history_plus_deletion`.

- [ ] **Step 3: Apply the same promotion gate**

If the current-pool result does not improve while preserving every gate,
record the negative result and make no production changes.

### Task 3: Implement only a promoted structural feature family

**Files:**
- Modify: `cvsearch/eval/replay_uncertainty_support.py`
- Modify: `cvsearch/eval/freeze_uncertainty_support.py`
- Modify: `cvsearch/eval/robust_transfer_selector.py`
- Test: `tests/test_replay_uncertainty_support.py`
- Test: `tests/test_robust_transfer_selector.py`

**Interfaces:**
- Produces: immutable prefix-history fields on `AggregateEvidenceFeatures`.
- Preserves: legacy policy loading and exact P0 fallback.

- [ ] **Step 1: Write a failing prefix-causality test**

```python
def test_structural_features_do_not_change_when_future_views_change(self):
    before = candidate_snapshots(stage2, split, calibration, policy)[0]
    mutate_only_views_after(split, before.observations)
    after = candidate_snapshots(stage2, split, calibration, policy)[0]
    self.assertEqual(before.evidence_features, after.evidence_features)
```

- [ ] **Step 2: Run and verify RED**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest -v \
  tests.test_replay_uncertainty_support \
  tests.test_robust_transfer_selector
```

Expected: failure because the promoted structural fields do not exist.

- [ ] **Step 3: Add only the promoted fields and shared feature mode**

Extend `AggregateEvidenceFeatures`, compute each field from the current prefix,
and add one shared feature-mode entry. Do not add dataset/backbone lookup or a
new action rule.

- [ ] **Step 4: Verify GREEN and exact nested reproduction**

Run the Step 2 tests and regenerate the five-partition nested report. Expected:
tests pass and metrics exactly match the promoted prototype within integer
counts and `1e-12` for mean observations.

- [ ] **Step 5: Commit or reject**

Commit production code only if exact reproduction passes every promotion gate;
otherwise revert only Task 3 changes and retain the diagnostic report.
