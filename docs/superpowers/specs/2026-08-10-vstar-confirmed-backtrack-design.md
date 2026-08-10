# B8 Vstar Confirmed-Backtrack Design

## Status and evaluation boundary

B5 improved the frozen Vstar development split from 33/37 to 34/37, but its
pre-registered complete-full rule reduced 171/191 to 167/191.  The complete
full result exposed only aggregate outcome statistics after commit `8082182`;
B8 must not inspect or route on complete-full labels, per-record correctness,
category, ordinal, option index, or image identity.  Because the aggregate
full result has been exposed, any later score on the same full benchmark is
classified as aggregate-exposed exploratory validation rather than a fresh
untouched test.

## Motivation

The paper assigns BACKTRACK a narrow role: recover a historical state after a
branch fails, then continue through a different unvisited branch and recompute
the answer.  B5 instead accepted any two-view or three-view local majority.
That made two correlated crops sufficient to replace the stable v2 answer.

B8 admits a replacement only for the trajectory pattern that directly tests
the paper's recovery claim:

1. the first ranked focus view agrees with the retained v2 P0 answer;
2. its selected SPLIT child disagrees with P0;
3. the disagreement triggers the already frozen B5 BACKTRACK observation;
4. the independent sibling branch agrees with the SPLIT child; and
5. the candidate passes the already frozen B5 development rule
   `p0.6666666666666666-c0.05-g-0.1`.

In shorthand the only admissible winner trajectory is `P0, C, C`, where
`C != P0`.  Two-view consensus, `C, C`, `C, X, C`, and every malformed or
infeasible trajectory retain P0.

## Invariance and leakage controls

The rule compares equality only.  It never checks whether an answer is option
0, a particular letter, or a particular semantic string, so applying any
bijection to every option index produces the correspondingly permuted result.
The same rule applies to every `logits_match` input and uses no benchmark or
question metadata.  Full B5 evidence was generated and hash-frozen before any
detailed full labels were opened.

## Development decision

On the frozen 37-topic Vstar development split, v2 scores 33/37.  The B8
trajectory selects one topic, corrects it, and corrupts none, for 34/37.  B7 HR
normalization is held fixed; B8 does not touch `option_list` outputs.

## Artifact and scoring protocol

The offline runner must validate the paired v2 inputs, every B5 file and
manifest against a source-embedded allowlist of precommitted freeze-report
hashes before deserializing B5 bytes, the exact disjoint union of B5
eligibility, record identity hashes, stored projections, and the frozen B5
decision.  Exact legacy and modern schemas reject unknown metadata.  It emits
all Vstar topics in source order plus a manifest containing the paired launch,
v2, and selected vector hashes.  The complete vector is committed before the
aggregate-only scorer is run.
