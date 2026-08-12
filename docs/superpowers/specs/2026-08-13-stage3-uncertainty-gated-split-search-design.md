# Stage 3 Uncertainty-Gated Split Search Design

## Decision status

Approved by the user's instruction on 2026-08-13 to prioritize accuracy on all
currently tested backbones and datasets and to proceed without intermediate
approval. Stage 1 v6 and Stage 2 v7 remain frozen baselines. Stage 3 is a new,
fail-closed layer and must not rewrite their configs, reports, ranking traces,
or baseline outputs.

## Scope and research question

The tested scope is exactly two backbones, Qwen2.5-VL-7B and InternVL2.5-8B,
and four datasets, V*, HR-Bench 4K, HR-Bench 8K, and TreeBench. The research
question is:

> Can a shared, answer-free uncertainty trajectory admit a bounded SPLIT and
> BACKTRACK search that produces new fine-grained evidence candidates, improves
> accuracy when results are pooled by every tested backbone and every tested
> dataset, and never lowers the point estimate in any backbone/dataset cell?

The independent variable is the addition of Stage 3 after the frozen v6/v7
pipeline. Primary dependent variables are official answer accuracy,
corrections, and corruptions. Secondary variables are candidate-oracle gain,
support calibration, false replacement rate, evidence visibility, MLLM calls,
processed pixels, and latency.

## Evidence motivating the change

The Stage 2 exposed transfer subsets contain 112 topics and 256 official answer
units. Candidate-oracle replay over P0, ZOOM, and EXPAND gives:

| Backbone | Dataset | Baseline errors | Existing candidate fixes |
|---|---:|---:|---:|
| Qwen | V* | 3 | 1 |
| Qwen | HR-Bench 4K | 13 | 5 |
| Qwen | HR-Bench 8K | 15 | 0 |
| Qwen | TreeBench | 7 | 1 |
| InternVL | V* | 2 | 0 |
| InternVL | HR-Bench 4K | 8 | 0 |
| InternVL | HR-Bench 8K | 19 | 0 |
| InternVL | TreeBench | 8 | 0 |

This separates two problems. Qwen HR-Bench 4K has a selection problem because
useful candidates already exist. Qwen HR-Bench 8K and every InternVL cell have
a candidate-generation ceiling, so selector or calibration tuning alone cannot
improve them. Stage 3 must generate genuinely smaller, query-relevant views.

## Alternatives considered

### Fixed-depth split

Always split the top patch and answer from the best child. This is simple, but
charges every topic and exposes already-correct answers to unnecessary changes.
It is rejected because the Stage 2 safety result depends on selective action.

### Uncertainty-gated split with backtracking

Split only when the evidence is relevant but insufficient, inspect a bounded
number of query-ranked children, confirm a candidate at two related scales, and
backtrack to the second branch when support falls or stalls. This is selected.
It directly addresses oversized patches while retaining P0 on incomplete or
conflicting evidence.

### General beam search

Maintain multiple branches to arbitrary depth. It has a higher theoretical
ceiling, but creates too many policy and budget choices before a one-branch,
two-depth search is validated. It is deferred.

## Architecture

### Frozen baseline boundary

Stage 3 consumes a complete Stage 1 v6 baseline row, its matching Stage 2 v7
observation row, the frozen backbone-selected support calibrator, and new
candidate-only SPLIT observations. It verifies the Phase 1 rank digest and P0
output before doing anything. Any mismatch returns the exact Stage 2 output.

Stage 3 never reads benchmark names, evaluator categories, target boxes,
answers, or correctness during inference. Answer type is an existing runtime
serialization contract, not a benchmark-routing signal.

### SPLIT candidate generator

The selected P0 focus patch is converted to an integer XYXY parent. Stage 3
implements the same frozen 2-by-2, 12.5%-overlap geometry contract as the
historical Phase 12 helper, but does not import evaluator code into inference.
A parity test compares all child boxes and stable identifiers against the
historical helper. No artificial super-resolution is applied; each child is a
native-coordinate crop.

All four children remain in the trace. Existing query relevance and the Stage 1
visual-information components rank them without pruning the recorded pool. Only
the top two children consume MLLM observation calls. Maximum search depth is two
and maximum visited branches is two. These constants are fixed rather than
exposed as tuning knobs.

Each visited child is rendered twice:

1. a tight native child crop;
2. a context-preserving padded crop around the same child.

Both views produce answer-free verbalized support and an answer under the
existing `logits_match`, `option_list`, or `option_single` contract. Candidate
observations are append-only and cannot replace P0 inside `perform_EGSearch`.

### Uncertainty trajectory

For each branch, the controller receives the calibrated support sequence
`(P0, tight_child, context_child)`. It records the Mann-Kendall S statistic over
the sequence and the exact raw/calibrated deltas. With only three observations,
S is used as a direction signal, not as a significance claim.

- positive direction plus sufficient final support admits confirmation;
- negative direction rejects the branch and backtracks;
- a plateau may backtrack when another child is unvisited;
- missing, malformed, or uncalibrated observations retain the frozen Stage 2
  output.

The query-derived evidence-demand vector remains a soft action prior. It may
order SPLIT versus context expansion, but it cannot override observed support
or confirmation.

### Answer replacement

An answer-changing candidate is eligible only when all conditions hold:

1. tight and context views have the same canonical answer;
2. both views are parseable and their final support passes the frozen shared
   threshold;
3. calibrated support improves over P0 and the trajectory is not decreasing;
4. the two views are distinct rendered images bound by hashes;
5. no other eligible visited branch supports the P0 answer with equal or higher
   selection score.

Otherwise Stage 3 returns the exact Stage 2 selected output. For HR `option_list`
rows, consistency is checked component-wise but a component changes only when
both candidate views agree; unconfirmed components retain their Stage 2 value.
This reuses the existing answer parsers and avoids a benchmark-specific rule.

### Backtracking and stopping

The first branch is the highest query-ranked child. A decreasing or plateaued
trajectory with insufficient evidence activates one BACKTRACK to the second
ranked child. Search stops after the first confirmed candidate, after two
branches, at depth two, or on the existing call/pixel budget. There is no
unbounded queue and no learned search policy in this stage.

## Components and interfaces

### `cvsearch/evidence_gap/split_search.py`

Pure inference helpers for child geometry, trajectory direction, branch
ordering, and fail-closed confirmation. The public helpers are
`generate_split_children`, `mann_kendall_s`, `rank_split_children`, and
`confirm_split_branch`. It may import answer utilities from
`cvsearch.evidence_gap`, but has no evaluator imports.

### `cvsearch/evidence_gap/types.py`

Adds immutable `SplitViewObservation`, `SplitBranchObservation`, and
`SplitSearchAudit` DTOs, plus an optional `split_search_audit` on `StepTrace`.
DTO validation binds boxes, render hashes, supports, answers, budget ledgers,
and the unchanged P0 anchor.

### `cvsearch/evidence_gap/method.py`

Adds four grouped config fields: `p5a_split_enabled`,
`p5a_split_replacement_enabled`, `p5a_split_render_policy`, and
`p5a_split_max_observed_branches`. The only admitted runtime configuration uses
candidate observation enabled, replacement disabled,
`native_2x2_overlap_two_scale_depth2_v1`, and at most two observed branches.
The method appends a SPLIT step after existing ZOOM/EXPAND observations. The
public CVSearch call signature and benchmark adapters do not change.

### `cvsearch/eval/replay_split_search.py`

Authenticates Phase 1/2 identity, applies the frozen calibrator, evaluates the
bounded trajectories through `select_split_candidate`, performs component-safe
answer projection, and emits one selection record. It is label-blind.

### `cvsearch/eval/score_stage3_transfer.py`

Evaluator-only paired scoring and candidate-oracle diagnostics. It may read
answers and boxes only after decisions and manifests are frozen.

## Experiment protocol

### Partitions

The already exposed Stage 2 transfer subsets become Stage 3 development data.
Before the first Stage 3 model inference, create deterministic disjoint
validation and final-holdout ordinal manifests from the remaining annotations.
The manifests bind annotation hashes, row hashes, model checkpoints, and the
frozen v6/v7 configs.

Development labels may select one shared support threshold and one shared
minimum gain from small declared grids. Backbone-specific support mappings are
the already frozen calibration outputs; no backbone-specific action threshold
is allowed. Validation results may reject a design but may not tune it. Final
holdout labels are opened once, after code, thresholds, and selection hashes are
frozen.

### Staged execution

1. Run CPU/unit tests and replay existing ZOOM/EXPAND candidates.
2. Run SPLIT candidate observation on exposed development rows for both
   backbones and all four datasets.
3. Measure candidate-oracle gain before tuning the selector. If a backbone or
   dataset has zero new oracle gain, change candidate generation rather than
   selection thresholds.
4. Freeze the shared selector on development data.
5. Run disjoint validation. Promote to final holdout only when every cell is
   non-regressing and accuracy pooled by every backbone and every dataset is
   higher than its frozen Stage 2 baseline.
6. Run the final holdout once and then, if compute permits, the complete
   benchmark partitions.

## Success gates

Stage 3 succeeds only if all applicable gates pass:

- Stage 1 v6 and Stage 2 v7 tracked artifacts remain byte-identical.
- Every observation has exact Phase 1 rank and P0 output identity.
- New SPLIT candidates add at least one candidate-oracle correction for each
  tested backbone and each tested dataset on development data.
- No backbone/dataset cell loses official correct units on validation or final
  holdout.
- Accuracy pooled across all datasets improves separately for Qwen and
  InternVL.
- Accuracy pooled across both backbones improves separately for V*, HR-Bench
  4K, HR-Bench 8K, and TreeBench.
- Total corrections exceed corruptions and aggregate accuracy improves.
- Answer-changing selections have two distinct agreeing rendered views and a
  non-decreasing calibrated support trajectory.
- Support Brier/ECE do not worsen for either backbone and AUROC is preserved.
- Calls, pixels, latency, branch count, depth, backtracks, and false
  replacements are reported; the fixed depth/branch/call budgets are never
  exceeded.
- Focused and complete unit tests, JSON validation, provenance checks, and raw
  log scans pass.

## Failure handling

Failures are attributed in this order: no focus patch, no new candidate-oracle
gain, child ranking miss, support miscalibration, trajectory selection error,
answer projection error, or budget exhaustion. Candidate-generation failures
may change split rendering or child ranking on development data. Selection
failures may change one shared threshold only on development data. A validation
or final-holdout failure is reported as such and is not repaired by reading the
same held-out labels.

## Limitations

- The scope is the two checkpoints and four datasets present in this workspace,
  not every possible MLLM or visual benchmark.
- Mann-Kendall direction over a three-point trajectory is descriptive and has
  low statistical power; no p-value claim is made.
- Strict improvement in every individual cell is a stretch goal. The promotion
  gate requires every cell to be non-regressing and every backbone-level and
  dataset-level aggregate to improve.
- Full-benchmark multi-step inference is substantially more expensive than the
  declared development and holdout runs, so bounded candidate-oracle screening
  precedes full execution.
