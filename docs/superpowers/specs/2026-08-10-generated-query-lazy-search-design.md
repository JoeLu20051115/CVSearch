# B5 Generated-Query Lazy Search Design

## Goal

Test whether changing the evidence path, rather than adding another answer
verifier or tuning its threshold, can produce a safer alternative to the
frozen v2 answer.  The implementation isolates the PDF's Main+Top-3 query
ranking, lazy `SPLIT`, and disagreement-triggered `BACKTRACK` mechanisms.

The acceptance condition is one benchmark-independent rule with strictly
positive development accuracy delta over frozen v2 on V*, HR-4K, and HR-8K.
Only then may the unchanged rule be evaluated on the sealed full partitions.

## Label boundary

Generation and selection receive only the validated base/combined rows:
`input_image`, original question, options for answering, and answer type.
Correct answers, boxes, targets, category, benchmark resolution, ordinal, and
correctness are forbidden policy inputs.  The localization-query call receives
only the original question; it never receives answer options or a candidate
answer.  Development labels are opened once after raw records, rule decisions,
complete output vectors, code, prompts, model artifacts, and pixel hashes have
been committed.  Full labels remain sealed unless a development rule qualifies.

## Query representation and ranking

`q0` is the unchanged question.  The frozen Qwen checkpoint receives one
text-only prompt requesting four short `LOC:` localization phrases that retain
objects, attributes, and relations but do not answer the question.  Parsing is
deterministic and requires at least one nonempty unique phrase.  Because a
generator can ignore the instruction and hallucinate an attribute value, a
deterministic sanitizer retains only words already present in `q0` plus the
semantically neutral location words `area`, `region`, `location`, `object`,
`image`, `visual`, and `evidence`.  Raw generated phrases are separately hashed
but never enter CLIP.  If no generated phrase survives, the sanitizer derives a
fallback only from non-function words already present in `q0`.

For every sibling patch, CLIP produces `Cmain` for `q0` and one score per
generated localization phrase.  `Caug` is the mean of that patch's top three
augmentation scores, or all available scores when fewer than three exist.
With fixed `beta=0.5`, `R = beta*Cmain + (1-beta)*Caug`.  Relevance, edge
density, and grayscale variance are converted to sibling-set percentiles.  The
existing combined score is retained: equal edge/variance visual weight and
`0.70` relevance plus `0.30` visual information.  Scores only order patches;
they never prune them.

## Lazy search path

The root sibling set is a deterministic overlapping 2x2 partition in original
image coordinates.  The highest-ranked root patch becomes focus and is
observed in the existing focus-plus-marked-context sheet.  Only then is that
focus split with the same overlapping 2x2 rule.  The highest-ranked child is
observed at higher spatial resolution.

If the parent and child produce the same canonical answer, search stops with
two-scale agreement.  If they disagree, `BACKTRACK` observes the highest-ranked
unvisited root sibling.  This branch is disjoint in the search tree even when
overlap causes a small pixel intersection.  No fourth answer view is allowed.

## Projection and rule grid

Two agreeing views form a candidate with path consensus `1.0`.  With three
views, an exact two-of-three canonical majority forms a candidate with path
consensus `2/3`; otherwise the candidate is infeasible.  V* uses summed option
losses and HR uses the existing semantic option-list aggregator.  Candidate
confidence is capped by path consensus.

Before label access, materialize every combination of:

- minimum path consensus: `2/3`, `1.0`;
- minimum candidate confidence: `0.5`, `0.75`;
- minimum confidence gain over P0: `0.0`, `0.1`, `0.25`.

No benchmark- or resolution-specific rule is permitted.  Among rules with
positive delta on all three development benchmarks, maximize the minimum
accuracy-point delta, then total corrected cycles, then use stricter path,
confidence, and gain constraints.

## Cost and failure behavior

Each topic uses one text-only Qaug call, two mandatory answer observations, and
at most one backtrack observation.  An HR observation contains four answer
calls.  Any generation, parsing, rendering, or inference failure retains the
frozen v2 decision and charges the planned per-topic budget.  Every source,
prompt, query, patch, render, observation, decision, model, and code artifact is
hash-bound.
