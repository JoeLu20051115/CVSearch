# Dense Query-Evidence Search Implementation Plan

> **Execution:** TDD first; freeze all nine decisions before development label
> access; do not inspect holdout/full labels unless one rule improves all three
> development benchmarks.

## Task 1: Tile bank, ranking, and rendering

- Add `phase9_dense_evidence_search.py` with pure 2/3/4-grid generation,
  exact deduplication, percentile ranking, and evidence-sheet rendering.
- Test geometry, border overlap, deterministic ties, answer-free rank inputs,
  panel pixels, hashes, source immutability, and invalid features.

## Task 2: Cross-tile projection and selector

- Add V* mean-loss/majority and HR four-shuffle semantic-majority projection.
- Add exact candidate DTO validation and all nine confidence/gain rules.
- Test thresholds, ties, forbidden metadata, nonfinite values, infeasible
  candidates, caller mutation, and exact P0 retention.

## Task 3: CLIP/Qwen producer

- Load frozen local CLIP and Qwen artifacts with explicit processor settings.
- Strictly validate phase-6 base/combined provenance and retain only v2-P0
  examples.
- Batch-rank 29 tiles, answer Top-3 sheets, and atomically write raw records
  plus a model/source/hash manifest under per-benchmark 512-call ceilings.

## Task 4: Three-GPU development run

- Run V*, HR-4K, and HR-8K concurrently on GPUs 0/1/2.
- Audit rows, failures, calls, hashes, candidate changes, and all nine rule
  counts without labels.
- Freeze a cross-benchmark decision table before scoring.

## Task 5: Development scorer and next stage

- Open development labels once and score all frozen rules plus candidate
  oracle. Select only a rule strictly improving all three partitions.
- If selected, commit the report and proceed to sealed full evaluation.
- If rejected, commit the negative ablation and use its oracle/failure pattern
  to design the next non-CVSearch candidate or verifier stage.
