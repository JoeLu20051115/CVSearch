# PDF-Faithful Cross-Backbone Evidence-Gap Search

## Status and scope

This document is the approved design for the method implemented after the
frozen Phase-15 cross-backbone batch.  It supersedes `minimal_v1` as the
research method, but it does not change or overwrite any Direct, original
CVSearch, or Phase-15 artifact.  Those runs remain immutable comparators.

The source of truth is `Query感知证据缺口引导的自适应视觉搜索.pdf`.
The non-negotiable search backbone is:

1. preserve the complete question as the CLIP main query;
2. produce answer-free query expansions and retain the best three CLIP
   expansion similarities per candidate;
3. combine main-query relevance, Top-3 expanded-query relevance, CVSearch
   feature complexity, and image edge density;
4. rank all candidates with that combined score and search in the resulting
   order;
5. repeatedly diagnose evidence gaps, execute actions, re-answer, verify,
   stop, or backtrack until a certified stop or an explicit forced return.

Every component is part of the default method.  An ablation can turn off one
component, but the production configuration cannot silently replace the full
method with native CVSearch order or a one-shot policy.

## Scientific contract

The primary baseline is the locally reproduced original CVSearch output using
the same backbone, checkpoint, benchmark protocol, image preprocessing, and
official evaluator.  `LOGIV-disabled` and Phase-15 are diagnostic controls,
not aliases for CVSearch.

If a component's matched ablation is negative, the first response is to audit
its embedding point, units, calibration, feasibility mask, action semantics,
and interaction with downstream gates.  Trace evidence must show whether the
component ran and whether it changed a decision.  A negative result cannot be
silently removed, renamed, or reported as success.  If the component remains
negative after an integration audit under a frozen protocol, both the result
and the audit are reported.

No benchmark label or evaluator-only metadata may enter query planning,
ranking, gap scoring, routing, verification, stopping, backtracking, or forced
return.  The policy view contains exactly `question`, `options`,
`answer_type`, and `input_image`.

## Alternatives considered

### Patch the current `minimal_v1` controller

This would reuse the largest amount of code, but that configuration explicitly
forbids SPLIT, EXPAND, and certified stopping, and its trace schema assumes a
single ZOOM and EXPAND audit.  Removing those constraints inside the same
runner would make old artifacts difficult to reproduce and blur the boundary
between the frozen proxy and the new method.

### Add a parallel faithful controller (selected)

Add a new controller and CLI while reusing tested CVSearch candidate hooks,
CLIP scoring, answer aggregation, and rendering primitives.  The new trace
schema is designed for a variable-length state machine.  Old commands and
outputs are unchanged.

### Rewrite CVSearch itself

This could express the policy directly inside the original recursion, but it
would mutate the baseline implementation and make paired attribution unsafe.
It is rejected.

## End-to-end data flow

For each sanitized sample:

1. Run the original CVSearch global sufficiency check and candidate-generation
   hooks without exposing labels to the policy.
2. Build `QueryPlan(q0, Qaug, E(q0))`.  `q0` is byte-for-byte the visible
   question.  The planner produces target-localization expansions and evidence
   requirements once, with deterministic decoding and strict JSON parsing.
3. If structured planning fails, use a logged deterministic fallback based on
   CVSearch's answer-free target extraction.  A fallback plan must still
   contain nonempty evidence requirements and at least one expansion; it may
   not fall back to native candidate order.
4. Adapt expert boxes and semantic-tree nodes to canonical candidates.  Build
   enough tree structure to expose children, but charge visual observation
   cost only when a state renders them.
5. At every newly revealed sibling set, compute and freeze the four raw score
   components and their within-sibling percentiles.  Insert every candidate
   into the global queue; no CLIP threshold may prune one.
6. Pop the highest-ranked unvisited candidate as the initial focus.  Render the
   focus plus any context and obtain an answer record.
7. Score the four independent evidence gaps.  Mask infeasible actions only
   after raw gap scoring, then execute the highest-scoring feasible action.
8. Recompute the answer record and independent evidence support for the new
   state.  Save a historical snapshot before making the next control decision.
9. Apply the ordered stopping/backtracking rules.  Repeat within the fixed
   cost budget.
10. Return either the certified current answer or the best historical answer
    with `FORCED_RETURN`; serialize the complete trace and the official
    evaluator-compatible output.

## Query planning

`QueryPlan` contains:

- `main_query`: unchanged `q0`;
- `targets`: normalized answer-free target descriptions;
- `augmented_queries`: distinct localization/detail/context expressions;
- `evidence_items`: target presence/detail, relation/context, and coverage
  requirements;
- `global_scope_required`: count, absence, uniqueness, or all-object scope;
- `planner_mode`, raw structured output, parse error, and fallback reason.

The planner must generate at least three useful expansions when possible.
Top-3 refers to the three highest CLIP similarities for each candidate, not the
first three generated strings.  Option text may be available to the answerer,
but expansions and evidence requirements must not assert or embed a candidate
answer.  Exact option-string leakage is rejected before ranking.

## Candidate ranking

For every sibling set `C`, and each candidate `c`:

- `m(c)`: CLIP similarity between the crop and `main_query`;
- `a(c)`: mean of the largest three crop-to-expansion similarities, or all
  available expansions when fewer than three survive validation;
- `v_c(c)`: CVSearch dense-feature complexity;
- `v_e(c)`: crop edge density.

Each vector is converted to an average-rank percentile within `C`.  Constant
vectors map to `0.5`.  The fixed fusion is:

`relevance = beta * pct(m) + (1 - beta) * pct(a)`

`visual = visual_lambda * pct(v_c) + (1 - visual_lambda) * pct(v_e)`

`rank = alpha * relevance + (1 - alpha) * visual`

The default uses the direct combined order from `rank`; it does not use
`ConservativeQueryRanker`, maximum displacement, or native CVSearch order.
Ties are resolved by native ordinal and canonical key.  Scores are frozen when
the sibling set is revealed so later search history cannot rewrite the queue.

Ranking assertions required in every full run:

- all four component arrays are finite and nonempty;
- every candidate appears exactly once before and after ranking;
- Top-3 aggregation is recorded per candidate;
- the popped order is monotonically non-increasing in frozen rank among
  currently available candidates;
- the ranking module reports how often it changed the native first choice.

## Search state and action semantics

`SearchState` contains focus candidates, context candidates, visited canonical
keys, visited render identities, revealed sibling groups, failed-action
signatures, remaining call/pixel/token budget, global ranked queue, current
answer/support records, and immutable history snapshots.

- `ZOOM`: render the same original-image coordinates at a higher level.  It is
  feasible only when its render identity is new.
- `SPLIT`: reveal the focus node's CVSearch children, rank that sibling set,
  and move to its highest-ranked unvisited child.  A leaf is an explicit no-op.
- `EXPAND`: keep focus and add the highest-ranked unvisited spatial neighbor
  that contributes new area or a missing relation/context item.
- `NEXT`: move to the highest-ranked globally available unvisited candidate.
- `BACKTRACK`: restore the best historical state with an unexplored branch.
  Consumed budget and visited render identities are never restored.

The controller never executes an unchanged action signature twice.  A no-op is
logged, masked, and the next feasible action is tried in the same iteration.
Every successful action must add a canonical observation, reveal new children,
or change the focus/context state.

## Evidence-gap controller

At each state, score four non-exclusive gaps in `[0, 1]`:

- missing local detail -> `ZOOM`;
- unresolved substructure -> `SPLIT`;
- missing relation or surrounding context -> `EXPAND`;
- likely wrong/insufficient region -> `NEXT`.

The prompt receives `q0`, `E(q0)`, the visible state, and a compact
answer-free action history.  It does not receive the candidate answer, ground
truth, `Qaug`, or evaluator metadata.  Strict JSON parsing produces all four
scores in one call.  If parsing fails, a deterministic analytic scorer uses
crop scale, available children, context coverage, target confidence, and queue
state.  The fallback is logged and still returns four raw scores.

Feasibility is evaluated separately.  Selection is argmax over feasible raw
scores with the stable tie order `ZOOM, SPLIT, EXPAND, NEXT`.  The trace stores
raw gaps, mask, masked order, selected action, and fallback mode.

## Answer uncertainty

V* uses three semantically equivalent answer prompts over the same option set.
The backbone produces per-option losses for each prompt; losses are averaged
before choosing the option.  Confidence records the normalized top-two margin,
prompt agreement, and all loss vectors.

HR-Bench uses its four shuffled cycles as semantic self-consistency.  Letters
are mapped to canonical option text, grouped by meaning, and projected back to
each official cycle.  The record contains raw outputs, canonical frequency,
margin, grouping, and aggregation availability.  If aggregation fails, the raw
cycle outputs remain eligible for forced return but cannot pass certified stop.

## Independent verifier

The full method uses a different frozen multimodal checkpoint from the answer
generator whenever the configured checkpoints differ.  The initial
cross-backbone runs use local Qwen2.5-VL-7B-Instruct as verifier for LLaVA-OV
and InternVL2.5 generators.  The runner rejects an accidentally identical
generator/verifier fingerprint.

For each evidence item, the verifier evaluates whether the current observation
visibly supports an answer-conditioned statement.  Yes/No logits are converted
to probabilities and recorded per item.  Coverage is additionally computed
mechanically from visited canonical regions.  `support_avg` and `support_min`
are separate gates; one strong item cannot hide an unsupported required item.

If the independent verifier fails on one sample, a logged CLIP-calibrated
support fallback may keep the search running, but the state cannot receive
`CERTIFIED_STOP`.  It remains eligible only for historical forced return.

## Ordered control and fallback closure

The decision order is fixed:

1. budget exhausted -> `FORCED_RETURN`;
2. all gap, uncertainty, support-average, support-minimum, and aggregation gates
   pass -> `CERTIFIED_STOP`;
3. progress stalled for the configured patience while support/confidence fails
   -> `BACKTRACK`;
4. execute the highest-scoring feasible evidence-gap action;
5. if it no-ops, mask it and try the next feasible action;
6. if every local action is unavailable, `BACKTRACK` to a historical branch;
7. if no historical branch exists, use `NEXT`;
8. if the ranked queue is empty -> `FORCED_RETURN`.

Historical quality is a predeclared label-free tuple over verifier minimum,
verifier average, answer stability, unresolved maximum gap, and cost.  Forced
return selects the maximum tuple and records the selected state id.  It is
never labeled certified.

## Implementation boundaries

Additive modules:

- `cvsearch/evidence_gap/pdf_types.py`: strict immutable DTOs and trace schema;
- `cvsearch/evidence_gap/pdf_controller.py`: pure transition, feasibility,
  stopping, backtracking, and forced-return logic with injected callbacks;
- `cvsearch/evidence_gap/pdf_runtime.py`: query planner, candidate adapter,
  model calls, render/action callbacks, and provenance;
- `cvsearch/perform_PDFSearch.py`: resumable CLI in a new output namespace.

Reuse rather than duplicate:

- `QueryAwareNodeRanker` and `fuse_scores` from `ranking.py`;
- `SearchStateCollector` candidate descriptors and render adapters;
- canonical answer aggregation from `answers.py`;
- original CVSearch hooks and official evaluators;
- existing model wrappers for LLaVA, InternVL, Qwen, CLIP, and SAM.

The new runtime cannot import an evaluator.  Existing Phase-15 modules cannot
be edited to call the new controller.

## Trace and module-activity contract

Every sample stores the query plan; raw and percentile ranking components;
native and combined ranks; queue pops; every state and action; four gaps and
feasibility masks; answer uncertainty; per-item independent support; history
quality; budget counters; fallback reasons; termination; and final provenance.

A full-run summary must report:

- structured-planner success/fallback rates;
- mean expansion count and fraction using a true Top-3;
- ranking coverage and native-first-choice change rate;
- invocation, feasibility, success, and no-op counts for every action;
- gap-score parse/fallback rates;
- verifier invocation/fallback rates;
- certified-stop, backtrack, and forced-return rates;
- mean MLLM calls, processed pixels, and steps.

The engineering gate requires every module to have a nonzero invocation count.
Natural smoke samples need not exercise every branch, so deterministic
synthetic controller fixtures must cover rare no-op, backtrack, certified-stop,
and forced-return paths.  Passing unit tests alone is insufficient: at least one
real sample must show combined ranking changing the native first choice and the
search following the combined order.

## Evaluation and ablation protocol

1. Keep the completed original Direct/CVSearch reproductions immutable.
2. Run CPU tests and one real sample per backbone/benchmark family.
3. Run a fixed label-blind smoke manifest and inspect only trace health, module
   activity, action diversity, fallback rate, and cost.
4. For each of Qwen2.5-VL-7B, LLaVA-OV-7B, and InternVL2.5-8B, run a fixed
   backbone-specific mini-development manifest against original CVSearch at
   matched budget.  The gate requires operational joint ranking, complete
   controller traces, no aggregate regression, and at least one auditable net
   correction.  Failure blocks full-scale evaluation for that backbone and
   triggers an adapter/scale/prompt/verifier integration audit; success on one
   backbone cannot waive another backbone's gate.
5. Freeze the full configuration before scoring benchmark correctness.
6. Evaluate LLaVA-OV-7B and InternVL2.5-8B with the official V*, HR-4K, and
   HR-8K evaluators.
7. Report absolute scores and deltas against local original CVSearch.  Because
   prior aggregate results are exposed, describe these as aggregate-exposed
   internal validation, not sealed confirmation.

Matched ablations use the same candidate pool and budget:

- native order vs main only vs main+Top-3 vs visual only vs full joint rank;
- full joint rank but shuffled search order, proving that rank is operational;
- fixed NEXT vs four-gap routing;
- no re-answering/re-scoring;
- no SPLIT, no EXPAND, no ZOOM, and no NEXT in separate runs;
- no independent verifier;
- support average only vs average+minimum;
- no backtracking;
- certified stopping disabled.

For each ablation, record both accuracy/cost and causal activity differences.
A module with identical traces to its disabled variant fails the embedding
gate even if aggregate accuracy happens to improve.

## Acceptance criteria

Engineering acceptance requires:

- strict policy sanitization and no label leakage;
- exact candidate preservation and direct joint-rank search order;
- active Main, Top-3, complexity, and edge components;
- variable-length multi-step traces with successful action-state transitions;
- separate answer uncertainty and independent support;
- exercised backtrack, certified-stop, and forced-return branches;
- resume/provenance rejection across code or configuration changes;
- exact output counts and unchanged official evaluator compatibility.

No backbone receives a full-scale run until its own fixed small-scale gate has
passed.  The small-scale result, manifest, and matched CVSearch outputs remain
part of the final report rather than being discarded after calibration.

Research acceptance is improvement over locally reproduced original CVSearch at
the same declared cost, or equal accuracy at lower declared cost.  Regardless
of that outcome, all full and ablation numbers are retained and reported.
