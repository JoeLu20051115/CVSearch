# Dense Query-Evidence Search Design

## Decision status

Approved as the next autonomous iteration by the user on 2026-08-10. This
stage is named B2. It follows the PDF's query-aware ranking, visual-information
ranking, budgeted observation, and post-action re-evaluation, while replacing
CVSearch region generation with an independent deterministic tile bank.

## Objective and sealed evaluation rule

The current best results are `171/191`, `616/800`, and `618/800`. B2 must use
one label-blind policy and improve every frozen development benchmark before
any holdout/full label is opened. Holdout/full outcomes are not used for
subsequent tuning. A full evaluation is published only if all three scores
strictly exceed the current best.

## Independent candidate bank

Given only the sanitized source image and question, generate all cells of
three grids: `2x2`, `3x3`, and `4x4` (29 candidates). Expand each cell by
12.5% of its own width and height on every side, clamped to the image. Remove
only exact duplicate pixel boxes. Candidate identity is the normalized grid,
row, column, and final pixel box. No CVSearch node, target box, target object,
answer, option content, label, category, resolution bucket, or evaluator state
may affect generation.

## Query and visual ranking

Use the frozen local OpenAI CLIP ViT-L/14 artifact. Average normalized text
features for three answer-free templates:

1. the exact question;
2. `visual region needed to answer: {question}`;
3. `visual evidence for: {question}`.

Options are not visible to the ranker. For every tile, compute CLIP cosine
relevance, grayscale edge density, and grayscale variance. Within each grid
level, convert the three measures to deterministic percentiles. Define visual
information as the mean edge/variance percentile and final rank score as
`0.70 * relevance_percentile + 0.30 * visual_information`. Sort globally by
score, relevance percentile, smaller normalized area, grid size, row, and
column. All candidates remain in the queue; the frozen observation budget
visits the first three.

## Observation and answer projection

For each visited tile, create one deterministic RGB evidence sheet:

- top: tile letterboxed into a `448x448` panel;
- bottom: source image with the exact tile box, letterboxed into `448x448`;
- fixed eight-pixel separator and the phase-7 background colour.

The two panels and final sheet are hash-bound and must be nonempty. V* runs
the exact multiple-choice loss prompt once per sheet. HR retains all four
option shuffles per sheet. Thus the maximum development costs are 3 calls per
V* topic and 12 calls per HR topic, excluding CLIP ranking.

Across three sheets, V* requires the mean-loss winner to match a strict tile
majority. HR first projects each sheet's four answers to one semantic answer,
then requires a strict semantic tile majority. Candidate confidence is the
minimum of majority frequency and aggregate answer confidence. Unavailable or
nonfinite aggregation is infeasible.

## Admission grid

B2 supplements only topics for which the frozen v2 selector retains P0. It
cannot veto an existing v2 action. A candidate must differ from P0 and satisfy
both inclusive conditions:

- candidate confidence at least one of `{0.50, 2/3, 0.75}`;
- candidate confidence minus P0 confidence at least one of
  `{0.00, 0.10, 0.25}`.

All nine rule combinations are computed from one raw observation set. The
same rule applies to both answer contracts and all datasets. The selector DTO
forbids benchmark, resolution, category, ordinal, target, label, correctness,
and evaluator metadata.

## Development and iteration

Freeze candidate rows, rule decisions, source/model/code/prompt/render hashes,
and the complete nine-rule table before opening development labels. A rule is
eligible only if its accuracy delta is strictly positive on V*, HR-4K, and
HR-8K. Maximize the minimum accuracy-point delta; ties maximize total corrected
cycles, then prefer larger minimum confidence and larger gain.

If no rule qualifies, reject B2 without inspecting holdout labels. Diagnose
the frozen development candidate oracle and proceed to a substantively
different PDF-aligned stage (generated localization queries, SPLIT/backtrack,
or an independent verifier), rather than tuning ordinals or full-set outcomes.

## Provenance and budgets

Bind source RGB, tile boxes, CLIP/Qwen artifacts, CLIP processor mode, rank
features, rank order, evidence sheets, questions by hash, options by hash,
raw observations, candidate projections, decisions, calls, pixels, and source
code. Each benchmark has a 512-Qwen-call development ceiling. Any mutation,
budget excess, model error, or render inconsistency fails closed to P0.
