# PDF-Faithful Cross-Backbone Controller Implementation Plan

> **Execution rule:** preserve every frozen Direct, CVSearch, and Phase-15
> artifact.  New runs go under `reproduction/pdf_faithful/`.

**Goal:** Implement and validate the PDF-faithful LOGIV controller on
LLaVA-OV-7B and InternVL2.5-8B with operational Main/Top-3 plus visual joint
ranking, multi-step evidence-gap actions, independent verification, stopping,
history, and fallback closure.

**Architecture:** Add a pure state-machine layer and strict DTOs, then inject
existing CVSearch candidate/render hooks and model adapters through a separate
runtime.  Reuse ranking, answer aggregation, provenance, and JSONL checkpoint
utilities.  Do not edit Phase-15 to call the new method.

**Stack:** Python 3.11, PyTorch/Transformers, local CLIP ViT-L/14, SAM 3,
LLaVA-OV, InternVL2.5, Qwen2.5-VL, unittest/pytest-compatible tests.

---

## Task 1: Freeze old-proxy results and lock the new configuration schema

**Files:**

- Create: `reproduction/pdf_faithful/configs/full_v1.json`
- Create: `tests/test_pdf_types.py`
- Create: `cvsearch/evidence_gap/pdf_types.py`

**Steps:**

1. Finish and score all running Phase-15 cross-backbone artifacts before
   changing research-method code.
2. Write failing tests for strict finite thresholds, mandatory enabled
   modules, distinct generator/verifier fingerprints, immutable candidate and
   state records, valid action/termination enums, and exact JSON round trips.
3. Run `python -m unittest tests.test_pdf_types -v` and confirm failure.
4. Implement the smallest strict DTO/config layer that passes the tests.
5. Add `full_v1.json` with Main, Top-3, complexity, edge, all four actions,
   re-answering, verifier, backtracking, and certified stopping enabled.
6. Run the focused tests and `git diff --check`, then commit.

## Task 2: Make direct joint ranking auditable and operational

**Files:**

- Modify: `cvsearch/evidence_gap/ranking.py`
- Modify: `tests/test_evidence_gap_ranking.py`
- Create: `tests/test_pdf_ranking_queue.py`
- Create: `cvsearch/evidence_gap/pdf_controller.py`

**Steps:**

1. Add failing tests proving Top-3 selects the highest three expansion
   similarities per candidate, all four percentile components contribute, no
   candidate is pruned, stable ties preserve native order, and queue pop order
   follows combined rank rather than CVSearch order.
2. Add a strict ranking-audit result around `QueryAwareNodeRanker` without
   changing its existing public behavior.
3. Implement the pure frozen ranked queue in `pdf_controller.py`.
4. Test native-first-choice change counters and rejection of duplicate or
   non-finite candidates.
5. Run:

   `python -m unittest tests.test_evidence_gap_ranking tests.test_pdf_ranking_queue -v`

6. Commit the passing ranking slice.

## Task 3: Implement the multi-step controller and complete fallback graph

**Files:**

- Modify: `cvsearch/evidence_gap/pdf_controller.py`
- Modify: `cvsearch/evidence_gap/pdf_types.py`
- Create: `tests/test_pdf_controller.py`

**Steps:**

1. Write deterministic callback fixtures for ZOOM, SPLIT, EXPAND, NEXT,
   no-op suppression, action re-scoring, stalled progress, BACKTRACK,
   CERTIFIED_STOP, budget exhaustion, empty queue, and historical forced
   return.
2. Confirm the controller tests fail before implementation.
3. Implement feasibility masking after raw four-gap scoring and stable argmax
   action selection.
4. Implement action signatures so an unchanged no-op cannot repeat.
5. Implement immutable history, branch availability, label-free state quality,
   non-refundable budget, and ordered stopping/backtracking rules.
6. Assert every successful transition changes observation/state and every exit
   is exactly certified or forced.
7. Run `python -m unittest tests.test_pdf_controller -v`, then commit.

## Task 4: Implement planning, candidate/action adapters, and answer records

**Files:**

- Create: `cvsearch/evidence_gap/pdf_runtime.py`
- Create: `tests/test_pdf_runtime.py`
- Modify only if an additive hook is required: `cvsearch/CVSearch.py`
- Modify only if an additive adapter is required:
  `cvsearch/evidence_gap/search_state.py`

**Steps:**

1. Write tests for strict policy sanitization, structured plan parsing,
   answer-option leakage rejection, deterministic nonempty fallback planning,
   canonical candidate preservation, sibling membership, child revelation,
   spatial expansion, render-level zoom, and global NEXT.
2. Implement structured Qwen query planning with deterministic decoding and a
   logged CVSearch-target fallback.
3. Adapt the existing collector descriptors to sibling groups and direct joint
   ranking; do not copy SAM/tree generation.
4. Implement V* three-prompt loss aggregation and HR four-cycle semantic
   aggregation through existing answer utilities.
5. Test that policy callbacks cannot see `bbox`, `target_object`, `answer`,
   category fields, or ordinals.
6. Run `python -m unittest tests.test_pdf_runtime -v` plus existing hook,
   ranking, answer, and search-state tests; commit.

## Task 5: Add four-gap scoring and a truly independent verifier

**Files:**

- Modify: `cvsearch/evidence_gap/pdf_runtime.py`
- Create: `tests/test_pdf_gap_and_verifier.py`
- Reuse: `cvsearch/eval/phase10_paired_verifier.py`
- Reuse: `cvsearch/models/modeling_dispatch.py`

**Steps:**

1. Write fake-model tests for one-call four-gap JSON parsing, independent raw
   values, post-score feasibility masking, analytic fallback, per-evidence
   support probabilities, mechanical coverage, and separate average/minimum.
2. Implement answer-free four-gap prompts and deterministic analytic fallback.
3. Load the local Qwen checkpoint on a distinct device/fingerprint as the
   LLaVA/InternVL verifier.  Reject identical generator/verifier checkpoints.
4. Implement per-item answer-conditioned Yes/No support.  A verifier fallback
   must mark the state ineligible for certified stopping.
5. Run focused unit tests, then a one-sample verifier smoke on an unused GPU.
6. Record measured peak memory; reduce batching before dropping independence.
7. Commit.

## Task 6: Build the resumable CLI and provenance boundary

**Files:**

- Create: `cvsearch/perform_PDFSearch.py`
- Create: `tests/test_perform_pdf_search.py`
- Reuse: `cvsearch/evidence_gap/io.py`
- Reuse: `cvsearch/evidence_gap/provenance.py`

**Steps:**

1. Write CLI tests for required generator/verifier/CLIP/SAM/spaCy paths,
   ordinal selection, chunks, strict config, resume/force exclusion, exact
   output schema, and code/config fingerprint rejection.
2. Implement runtime loading and inject planner, ranker, observer, gap scorer,
   answerer, and verifier into the pure controller.
3. Write outputs only to caller-provided new paths; never infer or overwrite a
   baseline path.
4. Add per-row trace validation before checkpoint write and exact-count
   validation on finalize.
5. Run focused CLI tests and existing IO/provenance tests; commit.

## Task 7: Prove module activity on synthetic and real smoke runs

**Files:**

- Create: `cvsearch/eval/pdf_trace_audit.py`
- Create: `tests/test_pdf_trace_audit.py`
- Create: `reproduction/pdf_faithful/smoke_manifest.json`

**Steps:**

1. Write audit tests for planner, true Top-3, four score components, ranking
   coverage, changed native first choice, every action count, gap fallback,
   verifier independence/fallback, re-answer count, backtrack, certified stop,
   forced return, cost, and malformed traces.
2. Implement strict per-row and aggregate trace audits.
3. Run deterministic synthetic traces to cover rare branches.
4. Run one V* and one HR sample per backbone; inspect trace health without
   correctness labels.
5. Run the fixed label-blind smoke manifest and require nonzero module
   invocation, operational rank order, valid outputs, and bounded fallback
   rates.
6. Freeze separate mini-development manifests for Qwen, LLaVA, and InternVL.
   Each backbone must show no matched-budget regression against its local
   original CVSearch outputs and at least one auditable net correction before
   any full-scale run for that backbone is admitted.
7. If one backbone fails, stop only that backbone's scale-up and debug its
   model adapter, score units, prompt/loss interface, and verifier calibration;
   another backbone's success cannot satisfy this gate.
8. If a module is invoked but causally inert, debug its adapter/calibration
   before any benchmark-scale run.  Record every correction in a new config
   revision.
9. Commit the audited implementation and frozen config.

## Task 8: Run matched ablations and full cross-backbone evaluation

**Files:**

- Create: `reproduction/pdf_faithful/configs/ablations/*.json`
- Create: `cvsearch/eval/pdf_scores.py`
- Create: `tests/test_pdf_scores.py`
- Generate: `reproduction/pdf_faithful/<backbone>/<benchmark>/*.jsonl`
- Generate: `reproduction/pdf_faithful/scores.json`

**Steps:**

1. Materialize strict configs for native/main/Main+Top-3/visual/full ranking,
   shuffled rank execution, fixed NEXT, no re-scoring, per-action removal, no
   verifier, average-only support, no backtracking, and no certified stop.
2. Run matched-cost smoke ablations first and use the trace audit to verify the
   intended single difference.
3. Freeze `full_v1` before reading correctness aggregates.
4. Run V*, HR-4K, and HR-8K on LLaVA and InternVL with resumable per-GPU jobs.
5. Run unchanged official evaluators and independently recompute scores in
   `pdf_scores.py`.
6. Compare the full method primarily to each locally reproduced original
   CVSearch result.  Report Phase-15 and disabled controls only as diagnostics.
7. Report accuracy, deltas, cost, stop modes, fallbacks, module activity, and
   every ablation, including negative results.

## Task 9: Final verification and handoff

**Steps:**

1. Run focused new tests, then the complete suite:

   `python -m unittest discover -s tests -v`

2. Separate pre-existing missing-artifact failures from regressions with exact
   test names and logs; no new failure is acceptable.
3. Run `git diff --check`, trace audit, exact row-count checks, independent
   scoring, and official evaluators.
4. Review the final diff for edits to baseline behavior and label leakage.
5. Use verification-before-completion before claiming success.
6. Deliver a single comparison table separating original CVSearch, frozen
   Phase-15 proxy, PDF-faithful full method, and matched ablations.
