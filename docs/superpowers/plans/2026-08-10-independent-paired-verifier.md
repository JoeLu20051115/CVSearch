# B3 Independent Paired Verifier Implementation Plan

**Goal:** Use a different frozen visual checkpoint to distinguish the existing
v2 P0 answer from B2's dense alternative without labels or resolution-specific
rules.

**Architecture:** Add a pure paired-support selector and a resumable verifier
runner. The runner reconstructs B2 sheets, converts both outputs to canonical
answer text, obtains final-token Yes/No support from Cosmos-Reason1-7B on the
same views, and materializes all six predeclared decisions with full
provenance.

---

### Task 1: Make Qwen-VL checkpoint dispatch configuration-driven

- Add a failing unit test showing that a path without `Qwen2.5` but with
  `model_type=qwen2_5_vl` loads the Qwen2.5-VL class.
- Replace path-name dispatch with a local `AutoConfig` model-type check.
- Run the focused model/support tests.

### Task 2: Implement the pure B3 verifier contract

- Create `cvsearch/eval/phase10_paired_verifier.py`.
- Test answer-text projection, paired aggregate validation, rule keys, sheet
  wins, average/minimum margins, strict fallbacks, forbidden metadata, and
  deterministic decisions before implementation.
- Keep the six-rule grid and prompt constants immutable.

### Task 3: Implement the label-blind GPU runner

- Create `cvsearch/eval/phase10_paired_verifier_runner.py` and tests with fake
  pairs, images, B2 rows, verifier results, failures, budgets, atomic output,
  and provenance.
- Validate and reconstruct B2 records and evidence sheets before verifier
  calls.
- Load Cosmos-Reason1-7B only after all input checks pass.

### Task 4: Verify and commit the implementation

- Run focused tests, the complete CPU suite, and one real two-answer/one-sheet
  smoke test on Cosmos.
- Audit the model artifact, source, prompt, output schema, and planned calls.
- Commit code and tests before generating development outputs.

### Task 5: Run B3 and freeze decisions

- Run one development benchmark per H200 GPU.
- Check process exits, row counts, six-calls-per-record accounting, source and
  sheet hashes, and output hashes.
- Recompute all six complete v2-supplemented decision vectors without labels,
  write a freeze report, and commit it.

### Task 6: Score once and continue

- Reopen only the already-designated development labels after the freeze
  commit.
- Select a rule only if all three deltas are strictly positive under the frozen
  objective.
- If qualified, run the sealed full evaluation. If not, commit the negative
  result and proceed directly to B4 localization-query plus SPLIT/BACKTRACK
  experiments without inspecting full labels.

