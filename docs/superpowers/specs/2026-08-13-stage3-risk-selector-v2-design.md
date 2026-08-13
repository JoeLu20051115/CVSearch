# Stage 3 Risk Selector v2 Design

## Status and scope

This design is approved by the user's instruction to continue changing the
Stage 3 decision layer without further approval. Stage 1, Stage 2, the fixed
sixteen SPLIT patches, CLIP-plus-visual ordering, answer-bearing observations,
and support probes are immutable inputs. Only STOP, CONTINUE, BACKTRACK,
REPLACE, and FALLBACK decisions may change.

## Objective

Convert substantially more of the forty already-observed oracle-fix units while
retaining fail-closed safety. The locked target is at least ten converted units,
no negative backbone/dataset cell, no Qwen HR4 corruption, positive gains on at
least three datasets and both backbones, at least one positive Qwen cell, and no
single-cell concentration of all gains. Fixed-observation, provenance, policy,
and decision hashes must verify exactly.

Development source-grouped out-of-fold replay is the only model-selection
surface. Opened locked labels may score a frozen policy but must not select its
features, thresholds, calibration, or transitions.

## Evidence-status correction after the first replay

The development-only v2 policy remains the clean source-grouped OOF result. Its
first replay on `validation_v3` exposed cross-partition support-scale inversion.
Because `validation_v3` labels had already been opened in earlier Stage 3b/v1
iterations, it is not a defensible unseen set. Two artifacts are therefore kept
separate:

- `opened_development_only`: selected exclusively by development OOF and usable
  for an honest regression failure decomposition;
- `opened_development_and_regression`: a partition-safe post-hoc refit that may
  demonstrate the decision layer's capacity but must never be reported as an
  unseen transfer result.

The post-hoc refit requires zero corruption and nonnegative cells separately on
both opened partitions. A new untouched observation split is still required
before making a broad-transfer claim.

## Diagnosis of v1

The v1 selector is safe but systematically conservative for three reasons:

1. A global raw-support floor treats backbone- and dataset-dependent support
   scales as comparable. This structurally rejects correct V* observations and
   pushes correct TreeBench candidates below the calibrated utility boundary.
2. A hard two-view rule makes agreement binary. It cannot distinguish a strong
   singleton awaiting confirmation from weak disagreement, and duplicates the
   role of calibrated risk.
3. Each checkpoint exposes only the plurality changed answer. A stable
   non-plurality answer disappears from the decision surface even when its
   support gain and conflict margin are better.

These are decision-layer problems; regenerating GPU observations is not needed.

## Candidate-independent evidence ledger

Every parseable changed canonical answer owns a ledger across distinct render
hashes. At each frozen observation checkpoint the selector materializes one
snapshot per changed answer, not only the plurality answer. A snapshot contains
only label-blind evidence:

- frozen P0 uncertainty;
- confirmation-aware agreement, `agreeing / max(2, parseable)`;
- calibrated support of the candidate evidence;
- candidate support gain over P0;
- candidate margin over the strongest observed P0 evidence;
- observed-view and distinct-render provenance.

All five numeric terms remain in `[0, 1]`. A single observation therefore has
at most `0.5` agreement but is not structurally forbidden. Raw support remains
a continuous feature through the frozen support calibrator; it is no longer an
empirical global veto. Unparseable output, duplicate render identity, malformed
rank/query provenance, altered geometry, or calibration mismatch still causes
exact P0 fallback.

## Hierarchical risk calibration

The raw uncertainty-support score remains a declared convex combination of the
five features. Development samples fit two regularized calibration heads:

- expected normalized correction benefit;
- expected normalized corruption risk.

The unified decision value is
`benefit - risk_penalty * corruption_risk`. Both heads are selected and refit
with source-grouped folds. Calibration first uses a backbone-and-answer-type
stratum when it has enough independent source groups, shrinks to the backbone
stratum otherwise, and finally falls back to the global stratum. Dataset names
are never inference inputs or routing keys.

When a linear/quadratic head cannot represent a bounded support-scale regime,
the same five-dimensional state may include a serialized axis-aligned
`RiskRegion`. A region is a calibration-tree leaf, not an alternate action
rule: it produces the same expected-benefit, corruption-risk, and margin values
consumed by the same state machine. Region selection is subject to the same
partition-level safety gates and is included in the authenticated policy.

Profiles, risk penalties, and replacement thresholds are finite declared grids.
Selection first enforces every development OOF cell/backbone/dataset safety
constraint, then maximizes official-unit corrections, coverage across datasets
and backbones, oracle-fix conversion, and finally minimizes corruptions and
observations. Stable declared ordering breaks exact ties.

## Unified state machine

For each revealed frozen view, update every candidate ledger and compute its
risk-adjusted value.

- `REPLACE`: choose the highest eligible candidate when its risk-adjusted value
  reaches the frozen threshold.
- `CONTINUE`: reveal the next view in the current branch while the same risk
  model's optimistic bound remains reachable.
- `BACKTRACK`: move to the next frozen branch when the bound is unreachable or
  the current branch is exhausted.
- `STOP_P0`: retain P0 when no candidate reaches the bound.
- `FALLBACK_P0`: return the exact Stage-2 output on any immutable-input,
  provenance, schema, hash, or calibration failure.

There are no dataset-specific rules and no alternate post-hoc selector path.

## Verification

Unit tests must first fail for independent candidate snapshots, soft singleton
evidence, hierarchical fallback, risk penalty, and legacy-policy loading. The
full existing suite must remain green. The freeze report records OOF groups,
coverage, corrections, corruptions, oracle-fix conversion, selected grid point,
and all input hashes. Final decisions are generated label-blind, then scored in
the existing 112-topic/256-unit table format with the user's target gates.
