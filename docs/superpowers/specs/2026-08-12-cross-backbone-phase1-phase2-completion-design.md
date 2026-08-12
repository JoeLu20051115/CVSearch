# Cross-Backbone Phase 1 and Phase 2 Completion Design

## Decision status

Approved by the user's instruction to continue without intermediate approval on
2026-08-12. The checked-in Phase 1 v1 files and report remain historical,
immutable evidence. New behavior is versioned as Phase 1 v2 and Phase 2 v3.

## Observed failures

On the same 30 V* topics, replaying the frozen Phase 1 v1 coefficients over
InternVL traces kept overall Recall@3 at 65.0%, improved direct attributes from
66.7% to 77.8%, but reduced relative-position Recall@3 from 63.6% to 54.5%.
The failure is therefore a component-weight transfer failure, not a candidate
pool failure. A label-blind grid over the declared development traces showed
that context questions need edge density rather than SAM feature deviation:
with one shared rule, Qwen Recall@3 becomes 100.0% and InternVL becomes 75.0%
on the development comparison while both categories improve or remain stable.

Phase 2's InternVL transfer failure is calibration overfit. Its raw verbalized
support already has lower held-out Brier and ECE than the 30-sample isotonic
mapping. Grouped cross-validation over monotone shrinkage weights selects 0.75
isotonic weight for Qwen and identity (0.0) for InternVL without reading held-out
labels. This preserves Qwen's HR gains and InternVL's V* correction.

## Phase 1 v2

Keep the v1 ranker defaults, configuration, report, and candidate pool. Add one
bounded coefficient, `context_visual_discount`, whose effective value is:

`effective_visual_lambda = clamp(visual_lambda - context_visual_discount * context_demand, 0, 1)`.

The promoted v2 configuration uses `context_visual_discount=1.0`. Detail-only
queries therefore retain feature deviation; context-demanding queries move
toward image-edge density; mixed queries remain governed by the same continuous
demand vector. The rule consumes only the sanitized question and query plan. It
does not read the backbone, benchmark, evaluator category, box, answer, or
correctness. Candidate identity and count remain unchanged.

The v1 configuration omits the new key. Its runtime values and trace schema stay
unchanged. A separate v2 configuration and report prevent retroactively
rewriting the original Phase 1 result.

## Phase 2 v3 calibration

Keep the support scalar, action policy, stop threshold, and action thresholds.
Add a frozen monotone calibration selector over weights
`(0, 0.125, 0.25, 0.5, 0.75, 1)`. For each weight, leave one source image out,
fit isotonic calibration on the remaining source groups, blend its prediction
with raw support, and score the omitted group. Select by mean Brier, then ECE,
then the smaller isotonic weight. Fit the selected mapping on the complete
calibration partition and hash the samples, groups, candidate metrics, selected
weight, and fitted knots.

Weight zero is an intentional identity-calibration outcome, not a failure. The
transfer calibration gate becomes: no backbone may worsen Brier or ECE beyond
floating-point tolerance, at least one backbone must improve both, macro Brier
and ECE must improve, and AUROC must be preserved. This is stricter about
cross-backbone harm than forcing every already-calibrated model to change.

## TreeBench support

TreeBench is the additional dataset gate. Extend candidate observation and
replay only for its existing `option_single` contract. Candidate answers use the
same prompt shape and official letter parser as CVSearch. The controller still
sees only normalized support, answer consistency, cost, demand, and action
availability. Evaluator-only target boxes are parsed from `target_instances`
after decisions are frozen to report geometric support and calibration; they
never enter inference or action selection.

## Experimental protocol

1. Development inputs are the existing Qwen and InternVL V* traces used to
   diagnose v1. They may select the single Phase 1 v2 coefficient and the
   calibration method family.
2. Held-out Phase 1 evidence uses previously uninspected InternVL spatial
   ordinals plus Qwen regression rows. Run the actual MLLM path, not only CPU
   replay, before promotion.
3. Regenerate Phase 2 observations on the promoted v2 ordering. Old v1
   observations remain diagnostic only and cannot establish v2 preservation.
4. Freeze model-specific calibration from source-grouped V* calibration rows,
   then evaluate V*, HR-4K/8K, and TreeBench without dataset-specific action
   thresholds.
5. Report every dataset/backbone cell, including neutral and failed cells. Do
   not claim universal generalization beyond the tested checkpoints and data.

## Acceptance gates

- Phase 1 v1 tracked files remain byte-identical to commit `c1ee03a`.
- Phase 1 v2 preserves every candidate identity and count.
- On each tested backbone, v2 Recall@3 is no lower than original CVSearch
  overall or within direct-attribute and relative-position groups, and at least
  one group improves.
- Actual Phase 1 answer accuracy does not regress on either tested backbone.
- Disabled Phase 2 v3 has zero output and rank-trace drift from Phase 1 v2.
- Per-backbone calibration satisfies the no-harm gate; macro Brier/ECE improve
  and AUROC is preserved.
- No tested dataset/backbone answer point estimate regresses, corrections are
  at least corruptions, and macro paired accuracy improves.
- TreeBench runs with the shared action policy and has complete, parseable
  paired outputs for the declared holdout subset.
- The complete unit suite, focused integrity checks, JSON validation, and
  `git diff --check` pass before completion is claimed.
