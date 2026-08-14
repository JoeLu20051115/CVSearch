# Candidate Visual-Entailment Diagnostic Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Test whether candidate-conditioned visual entailment supplies a
transferable safety signal that the rejected agreement, pairwise-comparison,
candidate-free, and structural families lack.

**Architecture:** Reuse the frozen observation-eight proposer. On the complete
source image, a shared Qwen2.5-VL-32B verifier judges one visible candidate under
two fixed support/contradiction/insufficient prompts. Both must support the
candidate. Every other outcome falls back to exact P0.

**Tech stack:** Python 3.11, Pillow, existing Qwen VLM loss interface, unittest,
single CUDA GPU.

## Constraints

- Use `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11`.
- Do not expose P0, labels, correctness, dataset, backbone, ordinal, category,
  or GT geometry to verifier inference.
- Use one shared agreement/confidence grid and exact P0 fallback.
- Charge two verifier calls; maximum feasible-topic observations are ten.
- First run only historical `validation_v2 × Qwen` at 4,194,304 pixels.
- Expand only if all four cells are nonnegative, net gain is positive,
  corrections exceed corruptions, and mean observations are at most 11.52.
- Do not access MME-RealWorld-Lite.

### Task 1: Add prompt, projection, and decision contracts

**Files:**
- Modify: `cvsearch/eval/pairwise_verifier.py`
- Test: `tests/test_pairwise_verifier.py`

- [ ] Write tests proving the prompts contain the proposal but not P0 or hidden
  metadata, and that both prompt variants must select support.
- [ ] Verify RED.
- [ ] Implement the smallest immutable projection and exact-fallback decision.
- [ ] Verify GREEN.

### Task 2: Integrate authenticated single-GPU collection

**Files:**
- Modify: `cvsearch/eval/pairwise_verifier_runner.py`
- Test: `tests/test_pairwise_verifier_runner.py`

- [ ] Write a runner test requiring the byte-identical whole image, two calls,
  candidate-only prompts, cost ten, and fallback on one uncertain vote.
- [ ] Verify RED.
- [ ] Add `candidate_entailment` as one manifest-bound evidence mode.
- [ ] Verify focused tests and the complete pairwise verifier suite.

### Task 3: Run the historical expansion gate

- [ ] Collect all 56 historical Qwen topics; only the 30 feasible proposals
  execute verifier calls.
- [ ] Authenticate output and manifest hashes and confirm zero failures.
- [ ] Replay every shared agreement/confidence grid point against exact P0.
- [ ] Record per-cell corrections, corruptions, net, and mean observations.
- [ ] Expand only if the predeclared gate passes; otherwise retain a negative
  report and stop this family.
