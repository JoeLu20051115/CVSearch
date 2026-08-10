# B4 Joint-Evidence Cross-Model Duel Plan

**Goal:** Replace the failed absolute Yes/No verifier with one threshold-free,
order-balanced, cross-model comparison of P0 and DENSE.

### Task 1: Pure joint renderer and duel selector

- Write failing tests for exact `2x2` geometry, panel/source immutability,
  order-swapped prompts, A/B logit decisions, Qwen loss agreement, ties,
  malformed values, and exact fallback.
- Implement the minimum pure renderer and unanimous selector.

### Task 2: Label-blind runner

- Write failing runner tests for A/B token provenance, three-call accounting,
  atomic failure, B2 binding, and exact decision materialization.
- Implement a runner that validates B2, reconstructs the sheet, loads both
  frozen models, and writes content-addressed records.

### Task 3: Verify, run, and freeze

- Run focused and complete CPU tests plus one real wrong-candidate smoke test.
- Commit implementation, then run the three development benchmarks on separate
  H200 GPUs.
- Reconstruct and commit the complete decision vectors without labels.

### Task 4: Score once and continue

- Score the already-designated development partitions only after the freeze.
- If all three improve, run sealed full evaluation. Otherwise commit the
  negative result and start B5 generated localization-query plus
  SPLIT/BACKTRACK work immediately.

