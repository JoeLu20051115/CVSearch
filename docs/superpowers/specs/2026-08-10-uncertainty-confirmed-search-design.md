# Uncertainty-Confirmed Search Design

## Decision status

Approved by the user on 2026-08-10. The method must remain fair on unseen
datasets: one label-blind rule is shared by V* Bench, HR-Bench 4K/8K, and
future multiple-choice image datasets.

## Objective

Improve the published action-calibrated result (`171/191`, `616/800`,
`618/800`) without selecting examples from full-set correctness and without
weakening an admission threshold merely to capture a known correction. The
strict V* engineering target is at least `174/191 = 91.10%`.

## Evidence motivating the change

The PDF requires both answer uncertainty and evidence support before an early
stop. The current quick gate can return a root answer even when its normalized
option-loss margin is low. A read-only comparison of already retained runs
shows one V* search state with stability gain above `0.75`; accepting only
that high-gain state changes no retained HR output in the same replay.

The existing phase-6 action pool has a V* oracle of `173/191`, so threshold
tuning alone cannot reach the strict target. Two additional action answers
are correct but have lower stability than P0. Blindly accepting them is not
valid: the same rule also accepts a known harmful zoom state. They therefore
require a new, visually distinct confirmation observation.

## Unified inference boundary

The inference producer receives only the sanitized question, option text,
image, model/search state, action geometry, and budget ledger. It must reject
and never serialize policy inputs containing benchmark name, resolution,
question category, ordinal, target box, target object, answer, label,
correctness, evaluator state, or split membership.

Development labels are opened only by a scorer after all candidate decisions
and their hashes have been frozen. Full-set labels are never used to tune a
threshold, prompt, rendering policy, or action rule.

## Stage A1: deferred SEARCH observation

Run a second, explicitly costed CVSearch observation with quick gate `0.8`
alongside the exact gate-`0.6` P0. The second run is a candidate producer, not
a replacement baseline. Both launches use the same model artifacts, input
partition, native CVSearch ordering, preprocessing, and action budget.

The SEARCH candidate is feasible only when:

1. the paired input and model provenance match exactly;
2. its output differs from P0;
3. its canonical stability is available; and
4. `candidate_confidence - p0_confidence >= 0.75`.

Otherwise SEARCH is absent and exact P0 is retained. The `0.75` boundary is
predeclared from the broad gap between high-gain search recovery and the
lower-gain HR changes in retained aggregate evidence; it is not refined on a
full-set grid.

## Stage A2: visually distinct action confirmation

Confirmation is attempted only for a feasible ZOOM or EXPAND candidate whose
answer differs from P0 and which was not already admitted by the v2 selector.
It cannot veto an already admitted v2 action.

The confirmation image is a deterministic two-panel RGB composition:

- top: the exact action candidate observation at native action resolution;
- bottom: the exact broader pre-action observation. If that observation is
  unavailable, use a letterboxed whole-image thumbnail;
- fixed eight-pixel separator and frozen background colour;
- no annotation boxes, target coordinates, answer text, or evaluator data.

The two panels must have distinct pixel hashes. A render no-op is infeasible.
The candidate and confirmation views are answered independently. For V*,
three fixed semantically equivalent question templates yield three loss rows
per view. For HR, the existing four option shuffles are retained and each
view yields one four-shuffle semantic projection. Prompts preserve the exact
question and options and never add a candidate answer.

An action becomes `confirmed` only when:

1. the original action winner differs from P0;
2. every canonical view projection agrees on the same candidate answer;
3. the aggregate projection is available and finite; and
4. the aggregate confirmation confidence clears a single development-frozen
boundary shared by actions and datasets.

The confirmation boundary is selected once from the frozen development
partitions. Candidate values are the coarse set `{0.25, 0.50, 0.75}`. Among
values whose point estimate is non-negative on all three development
partitions and strictly positive on at least one, maximize the minimum
development accuracy delta across the three partitions. Ties choose the
larger threshold. If no value qualifies, Stage A2 is rejected.

## Final selection

The candidate order is `SEARCH`, `EXPAND`, `ZOOM`; action name breaks only an
exact gain tie. Already approved v2 EXPAND/ZOOM admission remains unchanged.
Confirmed actions are additional candidates, not exceptions to v2 rules.
Among admitted states, choose maximum aggregate stability gain; if every gain
is non-positive after confirmation, retain P0.

No benchmark-specific thresholds, question-family routing, resolution rules,
ordinal exceptions, epsilon comparisons, or full-set retuning are allowed.

## Experimental protocol

1. Implement and unit-test pure DTO validation, aggregation, and selection.
2. Run Stage A1 and A2 only on the existing frozen development partitions.
3. Freeze code, prompt, render, config, input, model, and GPU hashes before
   opening any later evaluation outcome.
4. Evaluate V*, HR-4K, and HR-8K once with the same frozen policy.
5. Report development, aggregate-exposed engineering, and locked recovery
   evidence separately. A new external or secret-label dataset remains the
   required confirmation for a paper-level generalization claim.

## Stop rules

- Reject A1 if either HR development point estimate is negative.
- Reject A2 if no coarse confirmation threshold satisfies the joint rule.
- Do not compensate a rejection by adding dataset routing or lowering a
  full-set threshold.
- Preserve the published v2 bundles unchanged; every new bundle gets a new
  selector ID and commit-bound directory.
