# Cross-Backbone Full Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Direct, official CVSearch, and the promoted Phase-15 LOGIV_V2 method on LLaVA-OV-7B and InternVL2.5-8B across V*, HR-Bench 4K, and HR-Bench 8K.

**Architecture:** Preserve the released backbone wrappers and frozen selectors. Add a small common runtime dispatch plus the missing option-loss/text-only contracts, then run the existing Direct/CVSearch entry point and a model-neutral Phase-15 replay runner. Keep raw predictions model-scoped and compute all reported gains from local vectors.

**Tech Stack:** Python 3.11, PyTorch 2.7.1, Transformers 4.57.0, unittest, SAM 3, spaCy 3.8.7, CLIP-L/14, three NVIDIA H200 NVL GPUs.

## Global Constraints

- Do not retune official CVSearch settings or any Phase-15 selection rule.
- Do not alter the Qwen behavior or overwrite existing Qwen artifacts.
- Use deterministic decoding and the repository's official evaluators.
- Treat Phase-24 as a non-promoted negative result; Phase-15 remains the formal best.
- Store new outputs only under `reproduction/cross_backbone/`.

---

### Task 1: Isolated baseline and exact model snapshots

**Files:**
- Use: existing repository files without modification
- Produce: external Hugging Face cache snapshots for both target models

**Interfaces:**
- Consumes: commit `3e82e65` and the existing Python environment
- Produces: a clean worktree, passing baseline tests, and two resolved snapshot paths

- [ ] **Step 1: Create the isolated worktree**

Run `git worktree add /mnt/data3/data_xingrui/lueq/.worktrees/cvsearch-cross-backbone -b feature/cross-backbone-reproduction 3e82e65` after verifying the target is absent.

- [ ] **Step 2: Verify the clean baseline**

Run `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -v` in the worktree.

Expected: all existing tests pass before compatibility changes.

- [ ] **Step 3: Download and resolve both exact checkpoints**

Use `hf download lmms-lab/llava-onevision-qwen2-7b-ov` and `hf download OpenGVLab/InternVL2_5-8B` with the shared `/mnt/data3/data_xingrui/.cache/huggingface` cache.

Expected: each command returns an immutable `snapshots/<revision>` directory containing `config.json` and all indexed weight shards.

---

### Task 2: Add the common backbone contract with TDD

**Files:**
- Create: `cvsearch/models/modeling_dispatch.py`
- Modify: `cvsearch/models/modeling_llava.py`
- Modify: `cvsearch/models/modeling_internvl.py`
- Modify: `cvsearch/perform_EGSearch.py`
- Test: `tests/test_cross_backbone_runtime.py`

**Interfaces:**
- Consumes: a model snapshot path and device
- Produces: `load_search_model(model_path, device)`, and wrapper methods `multiple_choices_with_losses(...) -> tuple[int, list[float]]` plus `generate_text_only(prompt) -> str`

- [ ] **Step 1: Write failing dispatch and wrapper-contract tests**

Tests assert that config/path identity selects `ModelGlobalLocal`, `ModelInternvl`, or `ModelQwenVL`; that `multiple_choices_inference` delegates to the new loss-returning method; that finite loss values preserve the existing argmin winner; and that unsupported checkpoints fail explicitly.

- [ ] **Step 2: Verify RED**

Run `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_cross_backbone_runtime -v`.

Expected: FAIL because `modeling_dispatch` and the two wrapper methods do not exist.

- [ ] **Step 3: Implement the minimal contract**

Move no rendering logic. Reuse the exact loss loops already present in LLaVA and InternVL, convert detached losses to finite Python floats, and return `(argmin, losses)`. Keep `multiple_choices_inference` as a compatibility delegate. Implement text-only deterministic generation with each wrapper's native tokenizer/chat format. Replace only the Qwen-only branch in `_load_runtime` with `load_search_model`.

- [ ] **Step 4: Verify GREEN and regression safety**

Run the focused test, then the full unit suite.

Expected: focused tests pass and the full baseline remains green.

- [ ] **Step 5: Commit the compatibility layer**

Commit only the five files in this task with message `feat: support cross-backbone LOGIV runtime`.

---

### Task 3: Add model-neutral Phase-15 replay with TDD

**Files:**
- Create: `cvsearch/eval/cross_backbone_phase15.py`
- Test: `tests/test_cross_backbone_phase15.py`

**Interfaces:**
- Consumes: paired disabled/enabled evidence-gap JSONL files and their manifests, benchmark name, model, and CLIP snapshot
- Produces: one complete LOGIV_V2 prediction vector plus an audit manifest

- [ ] **Step 1: Write failing pure replay tests**

Use synthetic extracted pairs to assert: V* applies the frozen B8 confirmed-backtrack selector only after the existing Phase-6 decision retains P0; HR applies B7 semantic-majority normalization only when Phase-6 retains P0; otherwise the exact Phase-6 output is retained. Verify output order and cardinality are unchanged.

- [ ] **Step 2: Verify RED**

Run `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_cross_backbone_phase15 -v`.

Expected: FAIL because `cross_backbone_phase15` does not exist.

- [ ] **Step 3: Implement the smallest replay runner**

Reuse `validate_and_extract_combined_pairs`, `select_unified_state`, Phase-12 lazy rendering/projection, `select_vstar_confirmed_backtrack`, and `select_hr_semantic_normalization`. Replace the Phase-15 Qwen freeze trust-root check with a new-run manifest binding the actual backbone hash, config hashes, source JSONL hashes, selector source hashes, and output hash. Do not copy or modify selector rules.

- [ ] **Step 4: Verify GREEN and full regression safety**

Run the focused test and the complete unit suite.

Expected: all tests pass without changing existing Phase-15 publication hashes.

- [ ] **Step 5: Commit the replay runner**

Commit the runner and test with message `feat: replay Phase-15 across backbones`.

---

### Task 4: Smoke-test all three methods on both backbones

**Files:**
- Produce: `reproduction/cross_backbone/<model>/smoke/`
- Produce: model-scoped smoke logs

**Interfaces:**
- Consumes: exact model snapshots, local datasets/SAM/spaCy/CLIP, official and LOGIV runners
- Produces: six structurally valid one-record method outputs per benchmark family

- [ ] **Step 1: Run Direct and official CVSearch smoke records**

For each backbone, run ordinal 0 on V*, HR-4K, and HR-8K with Direct and official CVSearch settings on an assigned GPU.

Expected: valid V* integer output and four valid HR option responses with no traceback/OOM.

- [ ] **Step 2: Run disabled/enabled evidence-gap smoke pairs**

Use the frozen Phase-6 disabled and enabled configs for the same ordinals and verify matching run fingerprints, identities, and complete traces.

- [ ] **Step 3: Run Phase-15 replay smoke records**

Expected: one valid LOGIV_V2 output per benchmark/backbone, with hashes bound to the corresponding source vectors and model.

- [ ] **Step 4: Record the smoke gate**

Write a compact status JSON containing commands, exit codes, output schemas, and log anomaly scan results. Proceed to full runs only if all cells pass.

---

### Task 5: Run the complete paired experiment

**Files:**
- Produce: `reproduction/cross_backbone/<model>/<benchmark>/direct_answer.jsonl`
- Produce: `reproduction/cross_backbone/<model>/<benchmark>/cvsearch.jsonl`
- Produce: `reproduction/cross_backbone/<model>/<benchmark>/logiv_v2.jsonl`
- Produce: matching logs and manifests

**Interfaces:**
- Consumes: the passing smoke gate
- Produces: 18 complete method/benchmark/model prediction cells

- [ ] **Step 1: Run complete Direct and official CVSearch vectors**

Schedule independent benchmark jobs across GPUs 0/1/2, never placing two models on the same GPU concurrently. Preserve completed vectors and use resume only when the manifest matches exactly.

- [ ] **Step 2: Run complete disabled/enabled evidence-gap vectors**

Use the same model snapshot for each paired run and retain full launch manifests.

- [ ] **Step 3: Run complete Phase-15 replay vectors**

For V*, generate the lazy observations and apply B8. For HR-4K/8K, apply B7 to the validated Phase-6 pairs. Retain every intermediate hash and action count.

- [ ] **Step 4: Audit completion**

Expected: 191 unique V* rows per successful vector and 200 unique HR topics / 800 evaluator cycles per successful vector; all processes exit 0 and logs contain no traceback or CUDA OOM.

---

### Task 6: Score, independently verify, and report

**Files:**
- Create: `reproduction/cross_backbone/scores.json`
- Create: `reproduction/cross_backbone/report.md`
- Test: `tests/test_cross_backbone_scores.py`

**Interfaces:**
- Consumes: all complete prediction vectors and trusted annotations
- Produces: category scores, Overall scores, paired deltas, hashes, and verdict

- [ ] **Step 1: Write and run score-integrity tests**

Assert exact row/topic coverage, unique identities, valid option parsing, V* weighted Overall, HR cycle-level Overall, and agreement between official evaluator output and independent recomputation.

- [ ] **Step 2: Run official evaluators**

Record Attribute/Spatial/Overall for V* and FSP/FCP/Overall for HR-4K and HR-8K for all three methods and both backbones.

- [ ] **Step 3: Compute local paired gains**

For every model/benchmark compute `CVSearch - Direct`, `LOGIV_V2 - CVSearch`, and `LOGIV_V2 - Direct`; keep paper values in a separate reference field.

- [ ] **Step 4: Write the report and machine-readable scores**

Classify LOGIV_V2 as cross-backbone effective only if both backbones have complete local paired results; report mixed or negative outcomes without suppressing categories.

- [ ] **Step 5: Run final verification**

Run the full unit suite, JSON parsing checks, `git diff --check`, artifact hash verification, and a log scan for tracebacks/OOMs before claiming completion.

