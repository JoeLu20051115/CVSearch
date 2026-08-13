# Unified Uncertainty-Support Selector Design

## Decision status

Approved by the user's 2026-08-13 instruction to preserve the fixed Stage 3b
observations and replace the scattered selector rules with one unified
uncertainty-support state machine. Implementation may proceed without further
approval.

## Objective

Convert more already-observed correct SPLIT candidates without reintroducing
unsafe replacements. The new selector must use one shared decision model over
P0 uncertainty, multi-view agreement, calibrated support, support gain, and
conflict margin. It must control whether to stop, continue within a branch,
backtrack to another branch, replace P0, or fail closed.

This is a selector-only replay. It does not change the frozen sixteen-patch
geometry, CLIP-plus-visual ranking, query plans, model outputs, support probes,
view renders, branch order, or observation budget.

## Data and leakage boundary

Policy selection uses only the already-opened Stage 3b development observations
and labels. Cross-validation groups the same underlying source across both
backbones; HR-Bench 4K and 8K versions of one ordinal are also kept in the same
group. No fold may train on another rendering or backbone copy of its held-out
source.

The already-opened `validation_v3` suite remains useful as a locked regression
gate with the existing 112-topic, 256-official-unit accounting. Its labels may
not select profiles, calibrators, thresholds, feature definitions, or state
transitions. Results must call it a locked regression, not a new unseen claim.

Inference and replay never receive dataset names, evaluator categories, target
boxes, correct answers, or correctness bits. Backbone identity may select the
already-frozen support calibrator but may not select state-machine weights or
thresholds.

## Preserved components

The following effective Stage 3b components remain byte-identical inputs:

- exactly sixteen answer-free depth-two probes;
- the frozen `0.7 * CLIP + 0.3 * visual` ranking and path tie-breaking;
- the existing six answer-bearing branches and their fixed visit order;
- raw and backbone-calibrated evidence support;
- canonical V*, HR-Bench, and TreeBench answer projection;
- distinct-render SHA-256 checks, rank/query provenance, and fail-closed P0;
- the raw-support safety floor that prevents near-zero visual evidence from
  being promoted solely by calibration.

## Unified candidate representation

At every observation checkpoint, answers from distinct renders are grouped by
canonical answer. A changed candidate is structurally eligible only after at
least two distinct render hashes agree. For the leading changed group define:

- `u`: frozen P0 answer uncertainty in `[0, 1]`;
- `a`: agreeing-view count divided by parseable observed-view count;
- `s`: minimum calibrated support among the agreeing views;
- `g`: `s - p0_support`;
- `m`: `s - strongest_observed_p0_support`, falling back to `p0_support` when
  no separate P0-supporting view exists.

Normalize `g` and `m` from `[-1, 1]` to `[0, 1]`. A declared nonnegative weight
profile summing to one produces one raw advantage score:

`raw_advantage = w_u*u + w_a*a + w_s*s + w_g*(g+1)/2 + w_m*(m+1)/2`.

This is the only numeric decision score. Parseability, distinct render hashes,
provenance, and the existing raw-support floor remain structural safety checks,
not alternate selector paths.

## Source-grouped utility calibration

For each development candidate snapshot, let `delta` be its official-unit
correctness minus P0 correctness and `n` the topic's official-unit count. The
calibration target is `0.5 + delta / (2*n)`: harmful candidates are below 0.5,
neutral candidates equal 0.5, and helpful candidates exceed 0.5.

A deterministic monotone isotonic mapping converts raw advantage to calibrated
relative utility. Source-grouped out-of-fold replay compares only three fixed
profiles:

- `balanced`: `(0.20, 0.20, 0.20, 0.20, 0.20)`;
- `support_heavy`: `(0.10, 0.15, 0.35, 0.20, 0.20)`;
- `uncertainty_light`: `(0.10, 0.25, 0.25, 0.20, 0.20)`.

The shared replacement threshold is chosen from the declared calibrated
advantage grid `(0.00, 0.05, 0.10, 0.15, 0.20, 0.25)`, where calibrated
advantage is `2 * calibrated_utility - 1`.

A configuration is development-feasible only when every available
backbone/dataset OOF cell is nonnegative, every pooled backbone and dataset is
nonnegative, and corrections exceed corruptions. Among feasible configurations
select lexicographically by maximum net official-unit gain, fewer corruptions,
fewer answer-bearing observations, then stable profile/threshold order. Refit
the selected isotonic mapping on all development groups and bind its inputs,
profile, threshold, and payload by SHA-256.

## State machine

The replay consumes each frozen branch as a sequence of view observations.

1. `P0`: bind the original answer, uncertainty, and support.
2. `OBSERVE`: reveal the next already-recorded view in frozen order.
3. `CONTINUE`: when fewer than two distinct agreeing views are available and
   the raw-score upper bound can still reach the replacement threshold, reveal
   the next scale in the same branch.
4. `BACKTRACK`: when a branch is malformed, inconsistent, exhausted below the
   threshold, or cannot reach it even under optimistic bounds for unrevealed
   feature components, move to the next frozen branch.
5. `REPLACE`: stop on the first structurally eligible candidate whose calibrated
   advantage reaches the shared threshold.
6. `STOP_P0`: retain P0 after all branches are exhausted or every remaining
   candidate is unable to reach the threshold.
7. `FALLBACK_P0`: retain the exact frozen Stage-2 output on any provenance,
   schema, calibration, or budget failure.

Every transition records the state, branch, revealed roles, canonical candidate,
five input features when available, raw score, calibrated advantage, action,
and reason. Labels are absent from this trace.

## Alternatives rejected

A centralized hard-threshold selector would reduce code scattering but retain
the same brittle conjunctions and conservative behavior. A logistic model or
tree classifier could fit richer interactions but is unjustified for the small,
correlated development suite and would weaken interpretability. The selected
one-dimensional monotone utility calibration is the smallest learned component
that creates a genuinely unified decision variable.

## Validation and reporting

The locked regression report uses the previous table format:

- CVSearch/Stage-2 correct count and accuracy;
- unified selector correct count and accuracy;
- net official-unit change and percentage-point change;
- per-dataset pooled rows and per-backbone/dataset cells;
- corrections, corruptions, selections, action counts, mean observations, and
  exact input/policy/decision hashes.

The regression succeeds only when:

- Qwen HR-Bench 4K is at least its CVSearch baseline `33/48`;
- all eight backbone/dataset cells are no lower than CVSearch;
- all four pooled datasets and both pooled backbones are no lower than CVSearch;
- total correctness is strictly above `195/256` and corrections exceed
  corruptions;
- fixed-pool geometry, ranking, observation, provenance, and accounting audits
  pass exactly.

The current Stage 3b `201/256` result is a required comparison, not a hard
minimum. If development selects no feasible configuration or the locked
regression gate fails, the result must fail closed and report the exact state
and cell failure rather than tuning from validation labels.

## Components

- `cvsearch/eval/replay_uncertainty_support.py`: pure feature extraction,
  isotonic utility mapping, state transitions, and row replay.
- `cvsearch/eval/freeze_uncertainty_support.py`: source grouping, OOF selection,
  frozen policy creation, and deterministic CLI.
- `cvsearch/eval/score_uncertainty_support.py`: paired development/regression
  accounting and compact reports.
- Focused unit tests cover feature bounds, state transitions, grouped leakage,
  determinism, fail-closed behavior, HR atomic projection, and exact accounting.
