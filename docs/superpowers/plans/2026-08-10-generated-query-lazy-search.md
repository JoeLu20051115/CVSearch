# B5 Generated-Query Lazy Search Plan

1. Add failing pure tests for answer-free localization parsing, patch-local
   Top-3 fusion, contained/unique lazy children, backtrack routing, canonical
   two/three-view projection, and the frozen generic rule grid.
2. Implement the smallest pure `phase12` search module that makes those tests
   pass, reusing existing image, answer aggregation, and selector primitives.
3. Add failing runner tests for text-only query generation, exact call charging,
   source binding, lazy split, and conditional backtrack; then implement the
   GPU runner and immutable manifest.
4. Run focused tests, then the complete suite.  Commit code before inference.
5. Run V*, HR-4K, and HR-8K development partitions in parallel.  Without
   labels, replay every query, rank, patch, render, projection, and rule; freeze
   all complete output-vector hashes and commit the freeze report.
6. Open development labels once, score all frozen rules, and commit either the
   selected rule or a negative result.  Run the sealed full sets only if one
   unchanged rule improves all three development sets.
