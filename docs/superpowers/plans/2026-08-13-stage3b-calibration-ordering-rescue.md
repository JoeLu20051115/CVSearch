# Stage 3b Calibration and Ordering Rescue Implementation Plan

> Execute test-first and stop before full inference unless every frozen
> validation promotion gate passes.

**Goal:** Preserve the exact +7/zero-corruption Stage 3 v3 prefix, add a
development-only SPLIT calibration and bounded root-coverage rescue, then run
one new unseen two-backbone/four-dataset gate and promote immediately to full
only on 8/8 non-regression plus positive backbone and dataset aggregates.

**Architecture:** Replay v3 first on its original four branches. Only v3
fallback rows may use two appended root-rescue branches calibrated by a
SPLIT-specific frozen mapping. Runtime remains label-blind; evaluators alone
may read development boxes or post-freeze validation labels.

## Task 1: Lock the prefix and specify rescue contracts

**Files:**
- Modify: `tests/test_evidence_gap_split_observation.py`
- Modify: `tests/test_replay_split_search.py`
- Modify: `cvsearch/evidence_gap/types.py`
- Modify: `cvsearch/eval/replay_split_search.py`

- [ ] Add failing tests for six bounded branches, a four-branch v3 prefix, and
  exact v3 early return before rescue evaluation.
- [ ] Add failing tests that missing calibration, malformed rescue data,
  changed prefix identity, or a v3 selection returns the exact v3 result.
- [ ] Implement the smallest compatible schema/replay extension.
- [ ] Run focused tests and legacy Stage 3 replay tests.

## Task 2: Add all-root screening without changing the prefix

**Files:**
- Modify: `tests/test_evidence_gap_split_observation.py`
- Modify: `tests/test_evidence_gap_split_search.py`
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `cvsearch/evidence_gap/split_search.py`
- Add: `reproduction/evidence_gap/configs/dev_adaptive_ranking_observe_split_v4.json`

- [ ] Add failing fake-runtime assertions that the first eight probes and four
  branches are identical to v3, then append eight probes and one branch per
  omitted root.
- [ ] Add tight/medium/context render tests and require two distinct agreeing
  views for rescue confirmation.
- [ ] Implement stable all-root screening with fixed 16-probe/6-branch bounds.
- [ ] Verify exact Stage 2 runtime output and unchanged v3 prefix serialization.

## Task 3: Freeze SPLIT-specific calibration on development only

**Files:**
- Add: `cvsearch/eval/freeze_split_calibration.py`
- Add: `tests/test_freeze_split_calibration.py`
- Add: `reproduction/evidence_gap/adaptive_search_v7/split-calibration-manifest-v1.json`

- [ ] Add tests for development-only geometric labels, grouped selection,
  identity preference on ties, AUROC preservation, and no evaluator fields in
  frozen inference artifacts.
- [ ] Fit identity versus isotonic separately per backbone on opened V*/Tree
  development observations and freeze row/config/artifact hashes.
- [ ] Verify Qwen Brier/ECE no-harm and calibration provenance.

## Task 4: Development replay and policy freeze

**Files:**
- Add: `reproduction/evidence_gap/adaptive_search_v7/frozen-policy-v1.json`
- Add: `reproduction/evidence_gap/reports/stage3b-development.json`

- [ ] Run new rescue observations only on opened development topics.
- [ ] Report Recall@K and first-evidence rank with offline GT boxes, then strip
  boxes from inference/replay inputs.
- [ ] Freeze one shared rescue threshold policy only if v3 prefix identity is
  exact, corrections are positive, and corruptions are zero.
- [ ] Run strict JSONL, rank/P0 identity, configuration, and provenance audits.

## Task 5: Freeze and run the unseen promotion gate

**Files:**
- Add: `reproduction/evidence_gap/adaptive_search_v7/partition-manifest.json`
- Add: `reproduction/evidence_gap/reports/stage3b-validation.json`
- Produce untracked: `reproduction/evidence_gap/adaptive_search_v7/raw/validation/`

- [ ] Deterministically select 20/12/12/12 topics after excluding every prior
  manifest, bind annotation and row hashes, and commit before model inference.
- [ ] Freeze code/config/calibration/policy hashes and run both backbones over
  all four datasets once.
- [ ] Score only after all decisions are written; do not tune from validation.
- [ ] Require 8/8 non-regression, both backbone aggregates positive, all four
  dataset aggregates positive, aggregate gain positive, and all audits green.

## Task 6: Conditional full benchmark

**Files:**
- Add: `reproduction/evidence_gap/reports/stage3b-full.json`
- Produce untracked: `reproduction/evidence_gap/adaptive_search_v7/raw/full/`

- [ ] If and only if Task 5 passes, launch all 996 topics for both backbones in
  isolated result paths on safely available GPUs.
- [ ] Resume partial outputs by identity; never mix runner/config/model hashes.
- [ ] Score the complete run against Stage 2 and Stage 3 v3 and perform strict
  operational/provenance audits.
- [ ] Commit compact code, manifests, policies, and reports; leave raw model
  outputs untracked.
