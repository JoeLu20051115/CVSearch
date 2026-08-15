# Cross-Backbone Conservative Routing Design

## Decision status

Approved by the user's explicit instruction on 2026-08-15 to preserve the
current Qwen result and first make small, failure-driven routing changes for
InternVL and LLaVA without further intermediate approval.  The already scored
`full_v14` files are opened development evidence from this point forward; no
post-hoc result on those same files is an independent paper confirmation.

## Observed failures

The sealed `full_v14` comparison failed the cross-backbone gate.  InternVL was
net `-10/+14/-21` on V*/HR-4K/HR-8K and LLaVA was net `-1/+8/-2`.

Two causes are separable.

1. The PDF candidate factory forces `fast_threshold=2.0` so every sample emits
   a tree.  On native quick-answer rows this changes the answer used as P0.  The
   resulting forced-tree answer differs from the strict CVSearch file on 17
   InternVL HR-4K rows, 16 InternVL HR-8K rows, three LLaVA HR-4K rows, and two
   LLaVA HR-8K rows.  InternVL `insufficient_budget` fallbacks alone account for
   `-13` HR-4K and `-18` HR-8K units.  A safety fallback cannot be safe unless
   its anchor is the strict native answer.
2. The full PDF runner uses an unconstrained query-aware ordering.  On the
   evaluator-only V* center-hit diagnostic, native topic Recall@3 is 94.76% for
   both backbones, while the current combined order is 88.48% for InternVL and
   91.10% for LLaVA.  The candidate pool upper bound remains 100%, so this is a
   rank displacement failure rather than a tree coverage failure.

The full traces also show that support and paired margins alone do not separate
corrections from regressions.  Raising one global threshold is therefore not a
sufficient fix.

## Scope and invariants

- Qwen behavior, configuration, sealed MME result, and existing outputs remain
  unchanged.
- Only InternVL and LLaVA enter the new route.
- P0 is the output of strict family-native CVSearch thresholds.  Tree
  materialization is a separate operation and may never overwrite it.
- The full candidate identity pool is preserved.
- The native Top-3 set is preserved exactly.  CLIP and visual information may
  reorder only those three candidates in the first conservative version.
- Evaluator boxes and answer labels are diagnostics only and never enter the
  runtime route.
- Every fallback returns byte-for-byte the strict P0 object.

## Route A: strict P0 with lazy tree materialization

For InternVL and LLaVA, call CVSearch first with the exact native family quick
threshold (`0.6` and `0.8`, respectively) while collecting search state.  If
the result is not a quick answer, that call already supplies both P0 and the
tree.  If it is a quick answer, freeze its output as P0, then run the existing
`fast_threshold=2.0` path only to materialize candidates.  The second output is
audited but never used as the anchor or fallback.

Qwen retains the existing single forced-tree call in this branch so its frozen
path has zero behavior drift.

## Route B: protected-head CLIP and visual reranking

Wrap the existing `QueryAwareNodeRanker` with a three-candidate protected-head
policy.  It scores the complete pool, reranks only native positions 0--2, and
appends every remaining candidate in native order.  This makes Recall@3 and
Recall@5 invariant to reranking while still allowing Recall@1 improvement.

The opened full V* diagnostic selects the exploratory InternVL/LLaVA weights
`alpha=0.10`, `beta=0.70`, and `visual_lambda=0.70`.  Within the protected
native Top-3, the diagnostic changes Recall@1 from 62.83% to 78.01% on InternVL
and from 71.20% to 81.68% on LLaVA, while Recall@3 remains exactly 94.76% for
both.  These numbers are development diagnostics, not confirmation evidence.
The resolved route and weights must be included in each trace.

## Validation sequence

1. Unit-test strict P0 preservation on a simulated quick-answer row whose
   forced-tree output differs.
2. Unit-test that protected-head reranking preserves candidate identity, native
   Top-3 membership, tail order, and stable ties.
3. Unit-test that Qwen still uses its existing ranker and single candidate
   call.
4. Replay the saved V* traces evaluator-only and require no Recall@3 decrease
   overall or by test type, plus higher Recall@1 on both target backbones.
5. Run one label-blind smoke per backbone before any full inference.
6. Run InternVL and LLaVA on V*, HR-4K, and HR-8K.  These opened cells are
   development selection only.
7. Freeze the resulting policy before evaluating a new held-out or external
   confirmation partition.

## Deferred tree experiment

Replacing the connected semantic tree objective is a separate ablation.  Its
first version will rank flat, native-coordinate multi-scale proposals using
CLIP relevance plus visual information, select a patch, and build a local tree
only if later actions need descendants.  A second version may add
query-conditioned CLIP affinity to the connected clustering objective.  These
experiments are deferred until Routes A and B are measured, because they change
candidate coverage and would otherwise confound the anchor and ranking fixes.

