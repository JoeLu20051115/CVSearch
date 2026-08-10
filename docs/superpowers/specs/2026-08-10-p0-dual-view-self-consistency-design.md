# P0 Dual-View Self-Consistency Design

## Decision status

Approved by the user on 2026-08-10 after Stage A1/A2 produced no additional
admissions. This stage is named B1.

## Objective

Add answer candidates beyond the current P0/ZOOM/EXPAND pool while preserving
the published v2 results (`171/191`, `616/800`, `618/800`). The strict V*
engineering target remains `174/191 = 91.10%`; `173/191 = 90.58%` is reported
separately because it rounds to 90.6% but does not strictly exceed 90.6%.

## Scope and fairness boundary

B1 is selected by answer contract, not benchmark name. It applies only to
`logits_match` multiple-choice inputs. Existing HR four-shuffle aggregation is
unchanged. B1 may supplement only examples for which the frozen v2 selector
retains P0; it cannot veto an admitted SEARCH, EXPAND, or ZOOM state.

The producer may receive the sanitized image, question, options, validated P0
state, validated final search boxes, model artifact, and budget ledger. The
selector never receives benchmark, resolution, category, ordinal, target,
answer label, correctness, evaluator state, or split membership. Development
labels are opened only after all threshold decisions and hashes are frozen.
Full-set labels never tune prompts, views, thresholds, or routing.

## Frozen views

Reconstruct immutable non-root node descriptors from the validated P0 final
boxes. Every descriptor is non-root and uses the native CVSearch fine-search
rendering contract. Call the frozen Qwen renderer once:

- focus view: the renderer's zoomed evidence image;
- context view: the renderer's resized source image with its native focus
  boxes and arrows.

The crop intermediate is retained only for audit. Both selected views must be
RGB, nonempty, and pixel-distinct. Root-only P0 states and malformed/no-op
renders are infeasible and retain P0. No evaluator box, target annotation, or
label-derived geometry is admitted.

## Frozen prompts and aggregation

Both views use the three answer-preserving templates already frozen in phase
7:

1. the exact original question;
2. `Answer this visual multiple-choice question: {question}`;
3. `Using only visible evidence, answer this multiple-choice question:
   {question}`.

Options remain byte-identical and in their original order. Each call returns
one finite loss row. Per view, mean option loss determines the aggregate
winner and normalized top-two margin. The individual prompt winners must have
a strict majority for that same aggregate winner. View confidence is the
minimum of normalized margin and winner frequency.

## Admission rule

A B1 candidate is admitted only when all conditions hold:

1. v2 retained P0;
2. focus and context views are feasible and pixel-distinct;
3. both aggregate winners equal their own strict-majority winner;
4. both views agree on one output different from P0; and
5. `min(focus_confidence, context_confidence) >= threshold`.

The coarse threshold set is `{0.25, 0.50, 0.75}`. Threshold comparisons are
inclusive. Any missing, nonfinite, mismatched, or model-error observation
fails closed to exact P0.

## Development selection

Run all three thresholds from one frozen set of raw B1 observations. Because
the answer-contract routing leaves HR unchanged, the joint HR deltas are
structurally zero. Among thresholds with a strictly positive V* development
delta, choose the one with the highest V* development correctness; ties choose
the larger threshold. If none improves V*, reject B1 and proceed to B2 lazy
SPLIT candidate generation.

Before labels are opened, freeze code, prompt, render, model, paired input,
output, decision, and threshold-table hashes. Report candidate oracle only as
development headroom; it is never used as the selector.

## Cost and stop rules

Each feasible B1 example costs exactly six Qwen calls: three focus and three
context calls. The frozen development ceiling is 512 calls. Exceeding it aborts
the launch. B1 is not run on the full partition unless the development rule
selects a threshold. A failed B1 is retained as an audited negative ablation.
