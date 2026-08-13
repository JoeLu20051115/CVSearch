# Stage 3 Risk Selector v2 Implementation Plan

**Goal:** Replace v1's conservative hard gates with one candidate-independent,
hierarchically calibrated uncertainty-support risk state machine, without
changing any Stage 1/2 or frozen observation input.

**Constraints:** CPU replay only; development source-grouped OOF is the sole
selection surface; locked labels are scoring-only; exact fail-closed P0 and
legacy v1 artifact compatibility are required.

## Task 1: Freeze the behavioral contract

- Add tests proving every changed canonical answer receives a snapshot.
- Add tests proving singleton evidence is represented continuously rather than
  rejected by a hard two-view/raw-support rule.
- Add tests proving malformed provenance and duplicate renders still fall back.
- Run the focused tests and retain their RED output before implementation.

## Task 2: Implement candidate-independent replay

- Replace plurality-only snapshot construction with per-candidate ledgers.
- Make agreement confirmation-aware and keep all support terms continuous.
- Select and trace the best risk-adjusted candidate at each checkpoint.
- Preserve deterministic ordering, canonical projection, state traces, and v1
  policy deserialization.
- Run replay and historical split-search unit tests.

## Task 3: Implement hierarchical risk calibration

- Add monotone benefit and corruption-risk calibration heads.
- Fit backbone/answer-type, backbone, and global strata with deterministic
  source-group fallback and shrinkage.
- Add the declared risk-penalty and threshold grids.
- Prove held-out groups never enter their fold calibrators.
- Keep dataset names out of policy lookup and decision rows.

## Task 4: Select exclusively on development OOF

- Replay all fixed development rows through every declared configuration.
- Enforce nonnegative cell, dataset, and backbone OOF deltas before ranking.
- Rank feasible configurations by gain, coverage, conversion, corruption risk,
  observation cost, and stable grid order.
- Refit on all development groups and freeze the v2 policy with hashes.

## Task 5: Locked scoring and iteration boundary

- Generate locked decisions without loading labels.
- Verify decision/input/provenance/fixed-observation hashes before scoring.
- Score once against locked labels and emit the existing dataset/backbone table.
- Do not retune from a missed locked gate; use only an unused unseen observation
  split for another legitimate iteration.

`validation_v3` was found to be already opened by prior iterations. Preserve
the development-only failure unchanged, then label any repair fitted with those
labels as `opened_regression_postfit_not_unseen`. Such a repair can qualify the
implementation for a future unseen run but cannot itself establish transfer.

## Task 6: Regression, review, and integration

- Run the complete unittest suite.
- Audit the diff for label leakage, dataset routing, nondeterminism, and legacy
  artifact breakage.
- Commit the implementation and artifacts, merge locally into `main`, and
  report achieved gates and remaining evidence gaps with exact counts.
