# Unified Evidence-Gap Active Actions Implementation Plan

> **Execution:** use `superpowers:subagent-driven-development` task by task.  Each
> implementation task gets a fresh implementer, independent review, and at most
> one behaviour change before the next GPU ablation.

**Goal:** Add the PDF's active evidence-collection loop to the exact reproduced
CVSearch anchor, beginning with one `NEXT` observation, while preserving V* at
`171/191 = 89.53%` and requiring positive paired gains on both HR resolutions.

**Starting point:** HR uses exact gate-0.6 CVSearch raw output (`gamma=0`), V*
uses the verified root/search fallback, native CVSearch ordering is retained,
and no question-family, benchmark, resolution, category, ordinal, label, target
box, or target object may enter the controller.  Phase-1 `gamma=2.1` is not the
starting point because its exposed-full proxy changed HR-4K `613 -> 610` and
HR-8K `614 -> 611`.  Query reranking is also disabled because the earlier clue
was neutral on HR and negative on V*.

**Verifier status:** answer-free same-Qwen support and PDF-style
answer-conditioned same-Qwen support were both state-sensitive but failed the
joint HR replacement objective.  They may be traced for a genuinely new state,
but they cannot replace the anchor by themselves and cannot certify stopping.

## Frozen Phase-2A diagnostic rule

- Only `NEXT` is enabled; `ZOOM`, `SPLIT`, `EXPAND`, `BACKTRACK`, and
  `CERTIFIED_STOP` remain explicitly feature-disabled.
- P2A is an **all-feasible NEXT candidate-oracle run**.  It traces answer-free
  `s_gap` and `g_next=1-s_gap`, but does not claim that the rejected support
  diagnostic is a calibrated action controller.  No admission threshold is
  selected from the old root/search experiment.
- `NEXT` chooses the highest native-CVSearch posterior unvisited canonical
  candidate.  It never uses answer text, options, labels, or a question class.
- The exact P0 emitted answer is immutable and P2A always returns it.  NEXT
  candidate answers and fixed state features are audit-only.  An evaluator may
  calculate candidate-oracle headroom only after both raw outputs are
  materialized; oracle labels never enter the controller or candidate queue.
- P2B replacement is a separate reviewed commit and exists only if P2A shows
  positive candidate-oracle headroom on both HR resolutions while the full V*
  anchor remains `171/191`.  Its sole calibrated admission choice is frozen
  from the predeclared P2A protocol, not borrowed from the failed support
  selector.
- Same-Qwen answer-conditioned support remains a rejected proxy and is not
  called in P2A.  Its trace status is `disabled_same_checkpoint_unpromoted`.
  P2 always terminates as `FORCED_RETURN`; it cannot emit `CERTIFIED_STOP`.
- HR candidate diagnostics retain their original four `raw_outputs`; no hard
  canonical projection is introduced.  V* candidates keep the loss-based
  option winner.

---

### Task 1: Expose semantic-search state without changing CVSearch

**Files:**
- Modify: `cvsearch/CVSearch.py`
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `tests/test_evidence_gap_hooks.py`
- Modify: `tests/test_evidence_gap_method.py`

- [ ] Add trailing optional `search_state_sink=None` to public
  `get_cvsearch_response`, pass it from `get_evidence_gap_response`, and add
  trailing `search_state_sink=None`, `search_state_context=None` to
  `semantic_guide_search_dynamic_depth` without changing positional binding.
- [ ] Build search-state context independently of `node_ranker` for every
  main/cropped semantic-search call.  It contains `tree_scope`, original-image
  `crop_origin`, source-image identity, and a monotonic call ordinal incremented
  before each invocation.
- [ ] Emit `stage_ready` before each ordered queue is popped and
  exactly one `stage_finished` for every entered stage, including pop success,
  decay success, final-candidate success/failure, depth-one, empty-tree, and
  no-depth paths.  Live refs are process-local tuples; the separate snapshot
  must be strict JSON.
- [ ] Snapshot original-image `bbox`, `canonical_key`, parent/children, depth,
  stage, scope, crop origin, first rank, posterior, evaluated/selected/popped/
  remaining keys, and call ordinal.  Never serialize Node/PIL/model objects.
- [ ] Convert cropped-tree boxes to original coordinates before keying.  A sink
  may not mutate or replace the search queue.
- [ ] Canonicalize P0 `searched_nodes`, including fast/quick selections outside
  semantic events, so the collector can mark them visited.
- [ ] Use parent/child canonical original-coordinate keys, never local node IDs,
  even when an ancestor is outside the current depth.
- [ ] Test the full return-path matrix and all four context variants.  With
  P2 disabled the sink is never called and answer calls, ordering, return
  values, ledger, and output are byte-equivalent to current CVSearch.
- [ ] Run hook tests and full CPU suite; commit
  `feat: expose immutable CVSearch search states`.

### Task 2: Build a deterministic global NEXT queue

**Files:**
- Create: `cvsearch/evidence_gap/search_state.py`
- Create: `tests/test_evidence_gap_search_state.py`
- Modify: `cvsearch/evidence_gap/types.py`

- [ ] Collect first-seen live refs by canonical original-image key while keeping
  only JSON snapshots in the trace.
- [ ] Freeze a render contract containing original bbox and source-image
  identity.  When a cropped-tree candidate is selected, construct a render-only
  `NodeA(NodeState(original_image, bbox_original))`; never pass its local-bbox,
  cropped-image live node to original-image support/answer methods.  This is a
  rendering adapter for an existing candidate, not a proposal or queue change.
- [ ] Mark every popped/selected/current key visited, not only final boxes.
- [ ] Reject invalid/non-positive boxes and observations identical to any
  visited key.
- [ ] Order remaining nodes by finite posterior descending, then first-seen
  ordinal and canonical key; missing scores sort last.  Duplicate keys retain
  the first live ref.
- [ ] Return one candidate or a precise no-op reason; never create, prune, or
  rerank proposals with an auxiliary answer signal.
- [ ] Test main/cropped identity collisions, duplicates, ties, invalid boxes,
  nonzero-origin rendered pixels/geometry, no-op views, empty queues, and
  strict serialization; trace must never contain the render-only live object.
- [ ] Add immutable `P0Anchor(emitted_answer, cvsearch_raw, producing_phase,
  node_keys, support_view)` capture.  HR support view is the node bundle that
  produced raw CVSearch output; V* support view is the selected root/search
  bundle.  Missing/fast unrecoverable bundles make P2 a no-op, never a fallback
  to another view.
- [ ] Commit
  `feat: collect deterministic NEXT candidates`.

### Task 3: Add charged support and atomic observation batches

**Files:**
- Modify: `cvsearch/models/modeling_qwenvl.py`
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `cvsearch/evidence_gap/types.py`
- Create: `tests/test_evidence_gap_support.py`
- Modify: `tests/test_evidence_gap_method.py`

- [ ] Add a dedicated Yes/No-logit support adapter.  P2A gap mode accepts only
  `q0`, answer-free evidence requirements, a rendered observation, and its
  canonical identity.  Do not add or call the rejected same-checkpoint
  answer-conditioned verifier in P2A.
- [ ] Record prompt version/hash, processor mode, checkpoint, Yes/No token IDs,
  exact `p_yes` transform, requirement order, renderer geometry/view hash,
  logits/probabilities, requirement ID, avg/min, elapsed time, calls, pixels,
  and immutable batch-plan hash.  Reject contamination/non-finite payloads
  before the model call.
- [ ] Add a private `_BudgetedZoomModel.post_anchor_observation_batch` API.  It
  builds an immutable exact plan, pre-admits and consumes the ledger once, then
  invokes raw-model methods without the already-consumed terminal-answer
  wrappers.  The P2A plan includes current gap support, candidate gap support,
  and HR `4` or V* `1+len(options)` candidate-answer calls.  There are no
  answer-conditioned verifier calls, so the exact plan is known before any
  forward.
- [ ] No candidate support or answer call may run when the full batch cannot
  fit.  A raw-model exception remains fully charged, records a failed batch,
  and cannot replace P0.  Existing public terminal-reserve behaviour remains
  unchanged.
- [ ] Test support sanitization, exact call/pixel accounting, budget boundary,
  zero evidence items, explicit verifier-disabled status, processor provenance,
  raw-model exceptions, four HR
  outputs, every V* option count, and absence of partial execution;
  commit `feat: charge evidence support observations`.

### Task 4: Integrate one observation-only NEXT transition (P2A)

**Files:**
- Modify: `cvsearch/evidence_gap/method.py`
- Modify: `cvsearch/evidence_gap/types.py`
- Modify: `tests/test_evidence_gap_method.py`
- Create: `tests/test_evidence_gap_next.py`
- Create: `reproduction/evidence_gap/configs/dev_unified_next_oracle_gamma000_budget512.json`

- [ ] Add strict config keys `next_enabled`, `next_admission_mode`,
  `next_replacement_enabled`, and support prompt versions.  P2A requires
  `next_admission_mode="all_feasible"` and
  `next_replacement_enabled=false`; disabled mode requires zero behaviour
  change.
- [ ] Snapshot both the exact CVSearch raw response and the retained P0 emitted
  answer plus the exact producing support view before support/action work.
- [ ] Require at least one sanitized evidence requirement for gap support.
  P2A logs `uncertainty` from the existing immutable AnswerRecord,
  answer-free current/candidate gap support, normalized actual cost,
  `verifier_status="disabled_same_checkpoint_unpromoted"`, null verifier
  avg/min, and `coverage_status="not_observed"` separately.  It does not
  synthesize a Task-5 replacement score from missing support/coverage.
- [ ] Choose the deterministic NEXT candidate and atomically execute support
  and candidate answer.  Always restore the exact P0 object/output, including
  formatting; record `replacement_disabled_p2a`.
- [ ] Write `StepTrace(action=NEXT)` with current/candidate keys, `g_next`,
  supports, stability fields, score margin, feasibility, replacement reason,
  and budget; termination stays `FORCED_RETURN`.
- [ ] The config explicitly freezes mode/root tolerance/gate/budget/pixel
  accounting, HR fusion off/gamma zero, native rerank off, all non-NEXT action
  flags false, model/processor/prompt versions and hashes.
- [ ] Tests cover HR four-shuffle candidate capture, V* loss winner, zero
  requirements, support parse failure, missing P0 support view, no candidate,
  no-op, charged model exception, exact P0 return, and P2-disabled byte
  equivalence.
- [ ] Run focused/full CPU tests and independent review; commit
  `feat: add one unified NEXT evidence action`.

### Task 5: Run the three-dataset P2A candidate-oracle gate

**Files:**
- Produce untracked JSONL under `reproduction/evidence_gap/phase2_dev/v1/`
- Create: `reproduction/evidence_gap/reports/phase2-next-gpu-dev.json`

- [ ] Before running, freeze hashes for V* dev/holdout/full and paired grouped HR
  topic splits, source images, code, config, seed, Qwen/SAM/spaCy/processor,
  512-call/pixel limits, commands, and evaluator.  P0 and P2A use the same code
  revision and inputs.
- [ ] On three GPUs, run the same reviewed config and frozen Qwen/SAM/spaCy on
  the frozen V*, HR-4K, and HR-8K development partitions.
- [ ] Assert P2-disabled equals P0 first.  Then report paired accuracy, changed
  topics, corrected/corrupted, calls, pixels, latency, action feasibility,
  support deltas, replacements, and forced returns.
- [ ] P2A never promotes an emitted answer because replacement is disabled.
  Evaluator-only oracle analysis compares the already materialized P0 and NEXT
  raw candidates after inference.  Report paired topic bootstrap intervals;
  labels cannot choose the candidate, admission, or configuration.
- [ ] P2B is warranted only if candidate-oracle deltas are non-negative on both
  HR resolutions and their minimum is positive.  The full V* exploratory P0
  anchor must remain exactly `171/191` before any P2B HR full interpretation.
- [ ] If the gate fails, do not tune by question type or resolution.  Separate
  candidate-oracle failure (NEXT did not observe a better answer) from selector
  failure (a better state existed but was rejected/chosen wrongly).
- [ ] Commit only the compact reviewed report; never JSONL/logs.

### Task 6: Freeze P2B only with oracle headroom, otherwise choose the next PDF action

- [ ] If P2A candidate oracle is positive on both HR resolutions, predeclare one
  P2B admission rule from the frozen P2A protocol and implement replacement in
  a separate reviewed commit.  Require `n_requirements>=1`, non-worse answer
  stability, a uniquely defined canonical HR candidate (no tied top semantic
  groups), independently frozen verifier support, deterministic observed
  coverage (or omit coverage from the score), and strict positive fixed score
  margin.  Current/candidate ambiguity sets `verifier_unavailable`, makes no
  verifier call, and retains exact P0.  P2B must specify and test a new exact
  transactional budget plan for the post-answer verifier stage; it may not
  reuse P2A accounting implicitly.  Equal/missing evidence retains exact P0.
- [ ] P2B promotion requires V* non-regression, both HR deltas non-negative,
  positive joint minimum, and paired topic bootstrap reporting.  Do not tune a
  benchmark- or resolution-specific threshold.
- [ ] Select P2B exactly once on the named frozen development split, freeze its
  code/config hashes, then validate once on the separately named holdout; never
  cycle holdout outcomes back into threshold or module selection.
- [ ] If NEXT candidate oracle is non-positive, retain P0 and implement general
  coordinate-preserving `ZOOM` next, without the historical color/detail
  question gate.
- [ ] After a positive general ZOOM ablation, add lazy `SPLIT`; after SPLIT add
  `EXPAND` and `BACKTRACK`, one reviewed action per commit.
- [ ] Only after active actions are positive, evaluate a separately frozen
  verifier and `CERTIFIED_STOP` using PDF average/minimum support gates.  The
  same-Qwen diagnostic may not certify stopping.
- [ ] Create manifest v2 only after one configuration preserves V* and improves
  both HR resolutions.  Recovery/Vault remain unopened until that reviewed
  manifest binds code, configs, artifacts, budgets, commands, splits, and an
  aggregate-only scorer.

## Terminal acceptance

The engineering target is V* `>=89.53%`, HR-4K `>76.625%`, and HR-8K
`>76.75%` under the same unified controller.  After the first pass, continue
single-factor development tuning to maximize the minimum paired gain while
preserving V*.  Full public HR outcomes remain exploratory; a paper-level
claim requires the frozen recovery protocol or new external/secret labels.
