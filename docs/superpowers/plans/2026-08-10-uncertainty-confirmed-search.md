# Uncertainty-Confirmed Search Implementation Plan

> **Execution:** follow test-driven development and verify each task before
> moving to the next one.

**Goal:** Add one fair, dataset-agnostic deferred SEARCH candidate and a
visually distinct confirmation path for uncertain ZOOM/EXPAND candidates.

**Baseline:** selector v2 at `171/191`, `616/800`, `618/800`.

## Task 1: Pure confirmation DTOs and aggregation

- Create `cvsearch/eval/phase7_uncertainty_confirmation.py`.
- Create `tests/test_phase7_uncertainty_confirmation.py`.
- Start with failing tests for exact schemas, forbidden metadata, finite
  confidence, view agreement, render no-op rejection, and deterministic ties.
- Implement immutable `CandidateObservation`, `ConfirmationObservation`, and
  pure V*/HR aggregation helpers without label access.

## Task 2: Deferred SEARCH pair validation

- Extend the phase-7 module with strict paired gate-0.6/gate-0.8 validation.
- Test identical input/model/partition provenance, distinct run fingerprints,
  exact candidate output projection, and the fixed `>= 0.75` admission rule.
- Add an observation-only gate-0.8 config; do not alter the gate-0.6 P0 config.

## Task 3: Confirmation rendering and producer

- Add a deterministic two-panel renderer using exact action and pre-action
  observations, with whole-image fallback only when the pre-action panel is
  unavailable.
- Add three frozen V* equivalent-question templates and HR four-shuffle
  projection.
- Test pixel hashes, panel order, separator/background, no-op rejection,
  prompt hashes, budget charging, and fail-closed model errors.

## Task 4: Unified phase-7 selector and provenance

- Preserve v2 admission rules verbatim.
- Add SEARCH and confirmed-action candidates under the approved order.
- Bind selector source, rule material, prompt hashes, renderer hash, model
  artifacts, input manifests, and frozen development threshold into the
  decision digest.
- Update the final scorer to reject any mutation before label access.

## Task 5: Development experiments

- Run the exact frozen development partitions on three H200 GPUs.
- Evaluate SEARCH first. Stop A1 on a negative HR development effect.
- Evaluate confirmation thresholds `{0.25, 0.50, 0.75}` once from the same raw
  confirmation observations and apply the predeclared joint selection rule.
- Write an immutable development report with corrected/corrupted topics,
  cycles, calls, pixels, latency, and hashes.

## Task 6: Frozen three-benchmark evaluation

- Freeze one config and launch manifest before scoring.
- Run V*, HR-4K, and HR-8K once, one GPU per benchmark.
- Publish new selected bundles without overwriting v1/v2.
- Run the complete test suite, bundle hash audit, GPU/process audit, and score
  comparison against reproduction, paired P0, and selector v2.
