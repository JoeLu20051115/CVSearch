# B7 HR Semantic-Majority Normalization Design

## Goal and scope

B6 showed that cropped semantic option loss can produce useful alternatives, but
its replacement confidence does not transfer from HR-4K to HR-8K.  B7 does not
select a new semantic answer.  It fixes a narrower evaluator-boundary gap: when
the existing v2 P0 has already voted for one uniquely projectable semantic answer
in at least three of the four shuffled option blocks, emit that same semantic
answer's canonical letter in every block instead of preserving punctuation and
minority raw letters.

The rule applies only to `option_list` topics for which the frozen v2 selector
retains P0.  EXPAND/ZOOM decisions are unchanged.  `logits_match` remains fixed
to the previously developed B5 rule
`p0.6666666666666666-c0.05-g-0.1`.  Routing is by answer schema and v2 action,
never by benchmark, resolution, category, ordinal, or answer position.

## Projection and fail-closed behavior

Parse all four option blocks with the existing strict parser.  Map each official
letter extracted from the four raw P0 outputs back to canonical semantic text,
vote over semantic text, and use the existing deterministic first-vote tie rule.
Normalize only when all of the following hold:

- the winning semantic text has one unique reverse letter in every block;
- its vote frequency is at least `0.75` (a strict three-of-four majority);
- the projected four-letter output differs exactly from raw P0.

Malformed blocks, no valid votes, ambiguous winning projections, and frequencies
below `0.75` retain exact P0.  The threshold is the discrete strict-majority
boundary, not a benchmark-specific continuous fit.  No model call, image access,
annotation, candidate answer, or B6 loss is used by the HR rule.

## Evaluation boundary

Development labels had already been opened for B2-B6 and were used to identify
this rule.  B7 development results are therefore model-selection evidence, not a
new label-blind validation set.  The rule, implementation hashes, complete full
output vectors, and selected-topic identities must be frozen before any full HR
or Vstar labels are opened.

On development, the fixed combined rule must exceed the frozen v2 baseline on
all three benchmarks.  The sealed full run is successful only if it strictly
exceeds the current full best on all three: Vstar `171/191`, HR-4K `616/800`, and
HR-8K `618/800`.  A failure on any benchmark is recorded without adapting B7 to
full labels.
