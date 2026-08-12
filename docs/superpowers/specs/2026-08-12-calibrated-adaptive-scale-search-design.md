# Calibrated Adaptive-Scale Search Design

## Decision status

Approved in conversation on 2026-08-12. Phase 1 commit `c1ee03a` is an
immutable ranking baseline. Phase 2 may consume its ordered candidate trace but
must not change the Phase 1 ranker, query profile, candidate pool, or selected
ranking weights.

## Objective

Starting from the frozen Phase 1 Top-K order, choose among `ZOOM`, `EXPAND`,
`SPLIT`, `BACKTRACK`, and `STOP` using answer-free evidence requirements and a
calibrated support trajectory. Improve answer accuracy without increasing false
stops or hiding regressions behind one dataset, one resolution, or one MLLM
backbone.

## Scope and staged delivery

Phase 2A builds the model-neutral controller, calibration and metrics as pure
Python. It reuses existing observation audits and leaves emitted answers
unchanged until a selector passes the frozen development gate.

Phase 2B connects the existing Qwen `ZOOM` and `EXPAND` observation producers to
the frozen adaptive ranker and evaluates their candidate value. `SPLIT` remains
gated: it is implemented as an action decision and may be generated only when a
relevant parent is spatially ambiguous and neither `ZOOM` nor `EXPAND` resolves
the evidence gap.

Phase 2C moves support production behind the same adapter contract already used
by the repository's Qwen, InternVL, and LLaVA entrypoints, then runs a second
backbone and one additional benchmark. Model-specific calibration parameters
are allowed; benchmark-specific action rules and thresholds are not.

## Evidence-demand prior

The controller consumes a continuous demand vector rather than a hard question
class:

- `detail`: local appearance, text, number, texture, shape, or material must be
  visually resolved;
- `context`: multiple entities, a relation, or global coverage must coexist in
  the observation;
- `localization`: the relevant evidence may be mixed with distractors inside a
  coarse parent region.

The vector is inferred only from the sanitized question and query plan. It
initializes action preference and defines what support means. Its contribution
is bounded and decays after observations, so a strong support signal always
overrides it.

## Support observation and calibration

One normalized observation has:

- `p_full`, `p_partial`, and `p_none`, finite and summing to one;
- `support_consistency` and `answer_consistency` in `[0, 1]`;
- one missing reason from `none`, `detail_unreadable`, `context_missing`,
  `location_ambiguous`, `target_missing`, or `conflict`;
- normalized action cost.

The raw scalar is

`p_full * sqrt(support_consistency * answer_consistency)`.

A monotone isotonic calibrator fitted on a disjoint calibration partition maps
that scalar to `P(sufficient)`. Calibration labels describe whether the rendered
view satisfies the declared evidence requirements; answer correctness is
reported separately and is not silently substituted for support sufficiency.
The runtime uncertainty is `1 - P(sufficient)`.

## Action policy

The demand prior proposes an initial direction. The observed missing reason and
support trajectory determine the action:

- sufficient and stable support selects `STOP`;
- unreadable detail selects `ZOOM`;
- missing relation or neighbour context selects `EXPAND`;
- ambiguous location inside a relevant parent selects `SPLIT`;
- missing/conflicting evidence or two consecutive sub-threshold gains selects
  `BACKTRACK` when another branch exists;
- unavailable actions fail closed to the next feasible action and never delete
  a patch.

The controller receives no benchmark name, resolution, ordinal, evaluator
category, target box, correct answer, or correctness bit. A backtracked patch is
demoted in an append-only trace and remains recoverable.

## Model boundary

The controller consumes normalized support and answer records, not Qwen token
IDs or Qwen's historical `[-1, 1]` confidence. A model adapter is responsible
for rendering a view, returning an answer in the benchmark's answer contract,
and producing the normalized support DTO. HR option shuffles are canonicalized
to semantic answers before consistency is computed.

## Metrics

The primary endpoint is topic-paired answer accuracy against frozen Phase 1.
It is necessary but not sufficient. Every report also includes:

- corrections, corruptions, and false-stop rate;
- support AUROC, ECE, and Brier score before and after calibration;
- conditional answer and support gain for each action;
- backtrack recovery rate and trajectory length;
- MLLM calls, processed pixels, latency, and gain per added call;
- Phase 1 rank-trace identity and V* Recall@3 preservation;
- per-dataset and per-backbone results with a macro average.

## Experimental gates

1. Unit tests prove that disabled Phase 2 is byte-equivalent to frozen Phase 1
   and that the ranker files/config are not modified.
2. Development compares frozen Phase 1, hard routing, an uncalibrated one-step
   controller, and the calibrated trajectory controller under equal budgets.
3. A candidate controller must have zero Phase 1 rank-trace drift, lower ECE and
   Brier score than raw verbalized support, no increase in false stops, and no
   dataset point-estimate regression before holdout evaluation.
4. HR-4K and HR-8K versions of one semantic topic remain in the same partition.
5. After freezing prompts, calibrator, controller, thresholds, and hashes, run
   V*, HR-4K, and HR-8K. Add TreeBench or MME-RealWorld-Lite and at least one of
   InternVL2.5-8B or LLaVA-OneVision before making a cross-dataset or cross-model
   claim.

## Failure handling

Malformed or uncalibrated support, render no-ops, budget exhaustion, model
errors, and unavailable actions retain the Phase 1 answer. A failed development
gate triggers failure attribution by candidate generation, calibration,
selection, answer production, or cost; it does not authorize changing the
Phase 1 ranker or tuning a benchmark-specific exception.
