# B6 HR Semantic Option-Loss Design

## Goal and isolated change

B5 found one valid V* development correction but its free-form HR letters were
poorly calibrated.  B6 preserves B5's frozen answer-free Qaug, Main+Top-3
ranking, parent/child lazy split, and conditional root backtrack.  It changes
only the `option_list` answer projection: score the shared semantic choices by
option loss instead of generating a letter independently for every shuffle.

Routing is by evaluator-facing answer schema, never by benchmark, resolution,
category, or ordinal.  `logits_match` is fixed to B5 rule
`p0.6666666666666666-c0.05-g-0.1`, which scored 34/37 after the B5 freeze.
`option_list` uses the B6 rule grid described below.

## Semantic option reconstruction

Parse every HR option block with the existing strict parser.  All four blocks
must contain exactly the same four canonical semantic texts, each once.  The
first block fixes a deterministic semantic-choice order.  No annotation answer
is read.

For each B5 parent, child, and available backtrack sheet, call the frozen Qwen
checkpoint once with the original question and the four semantic answer texts.
The call returns one loss per semantic choice.  Parent and child are mandatory;
if their winners disagree, observe the already frozen B5 backtrack patch.

Two agreeing views give path consensus `1.0`; otherwise a two-of-three winner
gives `2/3`.  The winning semantic text is mapped back to its unique official
letter in every shuffle.  Confidence is the minimum option-loss margin among
the majority views, capped by path consensus.

## Selection and label boundary

Before B6 label access, freeze the Cartesian grid:

- minimum path consensus: `2/3`, `1.0`;
- minimum option-loss confidence: `0.0`, `0.05`, `0.1`, `0.25`.

The candidate must differ from frozen v2 P0.  No P0 confidence-gain comparison
is used because v2 HR confidence is a free-form shuffle frequency and is not on
the option-loss-margin scale.  One combined answer-schema rule qualifies only
if the fixed V* component and the same HR threshold pair strictly improve all
three development sets.  Full labels remain sealed until then.

Every B6 record binds the immutable B5 output/manifest hashes, reconstructed
semantic choices, original source pixels, replayed render hashes, Qwen
artifact, observations, projection, and all rule decisions.  Any mismatch or
failure retains v2 and charges the worst-case three-call topic budget.
