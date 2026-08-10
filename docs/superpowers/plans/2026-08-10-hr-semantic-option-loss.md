# B6 HR Semantic Option-Loss Plan

1. Write failing tests for strict cross-shuffle semantic reconstruction,
   semantic winner projection, two/three-view consensus, and the eight-rule
   selector grid.
2. Implement the minimal pure projection module; then add runner tests for one
   option-loss call per view and conditional backtrack.
3. Implement a runner that validates and re-renders frozen B5 HR evidence,
   performs only the new option-loss observations, and publishes immutable
   manifests.  Run focused and complete tests, then commit before inference.
4. Run HR-4K and HR-8K in parallel.  Replay all records, freeze eight combined
   complete-output vectors with the already fixed V* decision vector, and
   commit before opening HR development labels.
5. Score once.  If one same HR rule strictly exceeds 107/156 and 93/112, combine
   it with fixed V* 34/37 and run the sealed full evaluation.  Otherwise record
   the negative result and continue with a new verifier/candidate design.
