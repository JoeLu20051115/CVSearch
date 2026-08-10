# B7 HR Semantic-Majority Normalization Plan

1. Write failing tests for three-of-four normalization, exact four-of-four
   punctuation cleanup, two-of-four retention, ambiguous semantic projection,
   malformed input, and immutable P0 behavior.
2. Implement one pure projection/selection module by reusing the existing strict
   HR parser and semantic answer aggregator.  Run focused and complete tests.
3. Reconstruct the complete development outputs on top of frozen v2 decisions,
   bind them to source artifacts, and record the already exposed development
   scores for fixed B5 Vstar plus B7 HR.
4. Locate and validate full v2 combined inputs.  Run only the Vstar B5 evidence
   stage; apply B7 HR normalization without GPU inference.  Freeze every complete
   full output and provenance hash before opening full labels.
5. Score the sealed full outputs once.  Promote only if Vstar, HR-4K, and HR-8K
   all strictly exceed their current full best; otherwise record the result and
   continue with a new development-stage design without tuning to full labels.
