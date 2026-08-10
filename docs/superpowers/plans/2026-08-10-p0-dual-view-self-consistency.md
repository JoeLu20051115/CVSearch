# P0 Dual-View Self-Consistency Implementation Plan

> **Execution:** follow test-driven development and verify every label
> boundary before opening development labels.

**Goal:** Add a benchmark-blind P0 self-consistency candidate for
`logits_match` inputs without changing existing v2 admissions.

## Task 1: Pure B1 DTO and selector

- Add exact P0-view observation validation and immutable snapshots.
- Add three-row loss aggregation, strict-majority checks, conservative view
  confidence, dual-view agreement, and thresholds `{0.25, 0.50, 0.75}`.
- Start with failing tests for schema, forbidden metadata, view no-op,
  majority/aggregate disagreement, threshold boundaries, and caller mutation.

## Task 2: Native P0 view replay

- Reconstruct non-root descriptors from validated final boxes.
- Reuse the Qwen native node renderer and bind focus/context/crop pixel hashes.
- Test exact view selection, root/no-box failure, box validation, RGB mode,
  source immutability, and deterministic replay.

## Task 3: Label-blind producer and provenance

- Validate the paired phase-6 base/combined rows and retain only v2-P0 cases.
- Run three frozen prompts on each distinct view and write atomic JSONL.
- Bind prompt, options, model, runner, renderer, paired inputs, calls, pixels,
  and output hashes. Fail closed on per-example inference errors.

## Task 4: Development freeze and scoring

- Run the frozen V* development partition within the 512-call ceiling.
- Freeze all threshold decisions and hashes before label access.
- Score the three thresholds once; choose maximum positive development delta,
  tie-breaking toward the larger threshold. Otherwise reject B1.
- Record corrected/corrupted counts, candidate oracle, calls, and hashes.

## Task 5: Conditional full evaluation

- Only if B1 passes development, freeze one selector and run the full V*
  partition once. HR outputs are copied unchanged under the same selector
  identity because their answer contract is out of scope.
- Publish a new non-overwriting bundle and run the complete test/hash audit.
