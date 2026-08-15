# Cross-Backbone Conservative Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve Qwen while giving InternVL and LLaVA a strict native CVSearch anchor and a Top-3-safe CLIP-plus-visual patch order.

**Architecture:** The PDF runner separates native P0 generation from forced tree materialization only for InternVL and LLaVA.  A small ranker wrapper scores the full native pool but can only reorder the first three candidates, so the evaluated Top-3 set cannot regress.

**Tech Stack:** Python 3.11, unittest, Pillow, existing CVSearch/PDF runtime, existing CLIP scorer.

## Global Constraints

- Do not change Qwen runtime behavior or existing Qwen artifacts.
- Never read evaluator labels, V* boxes, or benchmark identity in runtime policy.
- Preserve every candidate identity and the exact native Top-3 set.
- Fallback output must equal the strict native CVSearch P0 object.
- Treat all six previously scored full cells as opened development data.
- Add no dependency and do not add untracked raw inference directories.

---

### Task 1: Strict native P0 and lazy candidate-tree materialization

**Files:**
- Modify: `tests/test_perform_pdf_search.py`
- Modify: `cvsearch/perform_PDFSearch.py`

**Interfaces:**
- Consumes: `family_candidate_kwargs(family: str, force_tree: bool = True)`.
- Produces: a native P0 for InternVL/LLaVA and an auditable forced-tree output digest when a second call is required.

- [ ] **Step 1: Write the failing threshold test**

Add assertions that `family_candidate_kwargs("internvl", force_tree=False)`
returns `fast_threshold == 0.6`, LLaVA returns `0.8`, and the default still
returns `2.0`.

- [ ] **Step 2: Run the threshold test and verify RED**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest tests.test_perform_pdf_search.PerformPDFSearchTest.test_candidate_factory_forces_tree_collection_but_keeps_family_thresholds -v
```

Expected: failure because `force_tree` is not accepted.

- [ ] **Step 3: Implement the minimum threshold switch**

Change the signature to:

```python
def family_candidate_kwargs(
    family: str, *, force_tree: bool = True,
) -> dict[str, Any]:
```

Use the existing family-native thresholds and set `fast_threshold` to `2.0`
only when `force_tree` is true.

- [ ] **Step 4: Verify GREEN**

Run the command from Step 2 and expect one passing test.

- [ ] **Step 5: Write the failing quick-P0 integration test**

Add a fake CVSearch function that returns `1` and `search_mode=0` at the native
InternVL threshold, but emits the existing fake tree and returns `0` at the
forced threshold.  Call `run_pdf_sample(..., generator_family="internvl")`
with a one-pixel budget and assert response `1`, two CVSearch calls, and a trace
whose native digest matches `1` rather than `0`.

- [ ] **Step 6: Run the integration test and verify RED**

Run its exact unittest name and expect the current forced-only path to return
`0` or make only one call.

- [ ] **Step 7: Implement lazy materialization**

For InternVL/LLaVA, run strict CVSearch with a collector first.  Reuse that
collector when `search_mode != 0`; only run the forced-tree call with a fresh
collector when `search_mode == 0`.  Keep the first output as `candidate_output`
for all comparisons and fallbacks.  Record the materialization output SHA-256
and whether a second call occurred in `candidate_factory`.

- [ ] **Step 8: Verify the integration and existing PDF tests**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest tests.test_perform_pdf_search tests.test_pdf_trace_audit -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add cvsearch/perform_PDFSearch.py tests/test_perform_pdf_search.py
git commit -m "fix: preserve strict cross-backbone p0"
```

### Task 2: Protected native Top-3 reranking

**Files:**
- Modify: `tests/test_evidence_gap_ranking.py`
- Modify: `cvsearch/evidence_gap/ranking.py`
- Modify: `tests/test_perform_pdf_search.py`
- Modify: `cvsearch/perform_PDFSearch.py`

**Interfaces:**
- Produces: `ProtectedHeadQueryRanker(base_ranker: Any, head_size: int)` whose
  call contract remains `(ranked_nodes, details)`.
- Consumes: the existing `QueryAwareNodeRanker` result and native candidate
  sequence.

- [ ] **Step 1: Write the failing protected-head ranker test**

Use a reversing base ranker over five identity-distinct nodes.  Assert that a
head size of three produces `[native2, native1, native0, native3, native4]`,
preserves all identities, and updates each detail's `combined_ordinal`.

- [ ] **Step 2: Run the ranker test and verify RED**

Run the exact new unittest and expect import failure because
`ProtectedHeadQueryRanker` does not exist.

- [ ] **Step 3: Implement the wrapper**

Call the base ranker once over the complete candidate sequence, map returned
details by node identity, keep only query-ranked nodes whose native position is
below `head_size`, and append the untouched native tail.  Rebuild details in
final order with consecutive `combined_ordinal` values.

- [ ] **Step 4: Verify the ranker test GREEN**

Run the exact new unittest and expect one passing test.

- [ ] **Step 5: Write failing family-route tests**

Assert that Qwen resolves the original configured `alpha/beta/visual_lambda`
without protected-head routing, while InternVL and LLaVA resolve
`0.10/0.70/0.70` with `protected_head=3`.

- [ ] **Step 6: Run family-route tests and verify RED**

Run the exact new tests and expect failure because the route helper is absent.

- [ ] **Step 7: Implement and trace the family route**

Add one small pure helper returning the resolved ranking policy.  Wrap only
InternVL/LLaVA rankers with `ProtectedHeadQueryRanker`.  Add the resolved policy
to the trace; leave Qwen's ranker construction byte-for-byte equivalent in
behavior.

- [ ] **Step 8: Run focused ranking/PDF tests**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest tests.test_evidence_gap_ranking tests.test_perform_pdf_search tests.test_pdf_trace_audit -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add cvsearch/evidence_gap/ranking.py cvsearch/perform_PDFSearch.py tests/test_evidence_gap_ranking.py tests/test_perform_pdf_search.py
git commit -m "feat: protect native top3 in cross-backbone ranking"
```

### Task 3: Development replay, smoke, and full gate

**Files:**
- Create: `reproduction/pdf_faithful/routing_v1/DEVELOPMENT.md`
- Create only after decisions finish: versioned manifests and compact score
  reports under `reproduction/pdf_faithful/routing_v1/`.

**Interfaces:**
- Consumes: sealed `full_v14` traces only for evaluator-side development
  diagnostics; new inference consumes images, questions, options, and frozen
  policy only.
- Produces: six complete label-blind decision files and a development score
  table relative to strict CVSearch.

- [ ] **Step 1: Replay V* ranking without inference**

Require for InternVL and LLaVA: candidate identity unchanged, topic Recall@3
equal to native overall and by test type, and Recall@1 strictly above native.

- [ ] **Step 2: Run complete unit verification**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python3.11 -m unittest discover -s tests -v
git diff --check
```

Expected: zero failures and a clean diff check.

- [ ] **Step 3: Run one label-blind smoke per target backbone**

Use the existing frozen model paths, one GPU per backbone, and one V* plus one
HR row when budget permits.  Audit every trace before expansion.

- [ ] **Step 4: Run the six opened development cells on three GPUs**

Schedule V*, HR-4K, and HR-8K by GPU and run InternVL then LLaVA in each lane.
Do not read scores until all six decision files and manifests are complete and
hashed.

- [ ] **Step 5: Score once for development selection**

Report all six cells, pooled results, corrections, regressions, runtime
failures, P0 materialization counts, and V* Recall@1/3.  Label the result
post-hoc development, not sealed confirmation.

- [ ] **Step 6: Commit and push the routed version**

Add only code, tests, design/plan documents, compact manifests, and compact
reports.  Do not add raw output directories unless explicitly selected for
publication.

