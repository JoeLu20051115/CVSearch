## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: pdf_faithful_mini_v16

## Fixed cross-backbone validation

- Runner: `pdf-faithful-v11-supported-cross-node-consensus`
- Git commit: `6da25b3`
- Code fingerprint in every result row: `253a2f77ca73f7e29a08ae5ac15a5c37abf15ccac0ee39e59e6becd3fd51c1d3`
- Frozen manifest: `reproduction/pdf_faithful/smoke_manifest.json`
- Units per backbone: 8 V* questions + 8 HR-4K shuffle decisions + 8 HR-8K shuffle decisions
- Acceptance gate per backbone: no regression and at least one net correction against the locally reproduced original CVSearch outputs

| Backbone | V* CVSearch | V* PDF | HR-4K CVSearch | HR-4K PDF | HR-8K CVSearch | HR-8K PDF | Total CVSearch | Total PDF | Corrections | Regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaVA-OV-7B | 5/8 | 5/8 | 8/8 | 8/8 | 7/8 | 8/8 | 20/24 | 21/24 | 1 | 0 |
| InternVL2.5-8B | 7/8 | 8/8 | 8/8 | 8/8 | 4/8 | 4/8 | 19/24 | 20/24 | 1 | 0 |
| Qwen2.5-VL-7B | 6/8 | 7/8 | 8/8 | 8/8 | 4/8 | 4/8 | 18/24 | 19/24 | 1 | 0 |

All nine JSONL files pass `cvsearch.eval.pdf_trace_audit --require-operational`.
The traces confirm whole-image-to-leaf trees, CLIP Main plus true Top-3 augmented-query ranking, complexity and edge-density ranking inputs, repeated re-answering, independent verifier calls, explicit tree actions, and certified fallback decisions. No independent-verifier fallback occurred.

The final answer-change gate requires a minimum state-evidence floor, a worst-case three-paraphrase contrastive margin above 0.10, non-root localization for pure target-detail questions, and support from at least two distinct focus nodes for history rescues. In v11, every focus node counted toward history consensus must itself pass the independent-verifier support floor without fallback. Relation questions retain the whole-image overview even when their plan also contains auxiliary target-detail evidence.

| Backbone | Result rows | Tree nodes | Ranked candidates | State evaluations | Gap-parser fallbacks | Planner fallbacks | Actions observed |
|---|---:|---:|---:|---:|---:|---:|---|
| LLaVA-OV-7B | 12 | 892 | 186 | 90 | 75 | 5 | BACKTRACK, SPLIT, ZOOM |
| InternVL2.5-8B | 12 | 892 | 120 | 87 | 4 | 1 | BACKTRACK, EXPAND, NEXT, SPLIT, ZOOM |
| Qwen2.5-VL-7B | 12 | 763 | 84 | 83 | 0 | 6 | BACKTRACK, EXPAND, NEXT, SPLIT, ZOOM |

The LLaVA action-score JSON parser used its explicit analytic fallback in 75/90 states; this is visible rather than hidden. The uncertainty re-answering, ranking, tree traversal, and independent verifier remained active in those states. This parser robustness gap should be reported separately from answer accuracy.

## Failure disclosure and repair

The stopped v10 full-V* prefix exposed one additional InternVL regression at ordinal 28 after 30 rows. A second focus node was counted toward history consensus even though its independent-verifier `support_min` was 0.453, below the configured 0.5 floor. The partial v10 outputs are retained under `reproduction/pdf_faithful/full_v10/` as failure evidence.

v11 makes the consensus count support-aware. The targeted v11 check on InternVL ordinals 28 and 116 changes the matched score from CVSearch 1/2 to PDF 2/2: ordinal 28 is retained by `proposal_lacks_spatial_consensus` with a supported-node count of one, while ordinal 116 remains a genuine correction. The Qwen ordinals 19 and 132 check likewise changes 1/2 to 2/2, and LLaVA HR-8K ordinals 12 and 15 change 7/8 to 8/8. All three targeted files pass strict trace audit.

## Validation boundary

This is a fixed small-scale engineering gate, not a paper-level full-benchmark estimate and not a statistical significance claim. The selected HR aggregates were previously exposed in this workspace. Earlier exploratory prefixes and targeted failures informed generic verifier and localization rules; they are not pooled with the fixed mini results. Full v11 runs are reported separately when complete.

## Artifacts

- `reproduction/pdf_faithful/mini_v16/llava/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/mini_v16/internvl/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/mini_v16/qwen/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/smoke_v16/internvl/vstar_guard28_correction116.jsonl`
- `reproduction/pdf_faithful/smoke_v16/qwen/vstar_guard19_correction132.jsonl`
- `reproduction/pdf_faithful/smoke_v16/llava/hr8_12_15.jsonl`
