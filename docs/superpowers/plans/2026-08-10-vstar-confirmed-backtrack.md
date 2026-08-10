# B8 Vstar Confirmed-Backtrack Plan

1. Add failing pure-selector tests for exact `P0, C, C` admission, all
   fail-closed trajectories, malformed records, and option-permutation
   equivariance.
2. Implement the smallest immutable B8 decision module that reuses the frozen
   B5 decision and observations.
3. Add failing runner tests for multi-file B5 partition union, provenance
   mismatch, trusted pre-exposure freeze authentication, missing or duplicate
   records, and complete output hashing.
4. Implement the offline runner and reproduce 34/37 on development data.
5. Run the full test suite, generate B8 from the already frozen full B5
   artifacts, independently replay every decision, and commit a no-label
   complete-vector freeze.
6. Query only aggregate full correctness.  Promote B8+B7 only if Vstar exceeds
   171/191 while the already frozen B7 HR results remain 620/800 and 628/800;
   otherwise record the result and continue development-only iteration.
