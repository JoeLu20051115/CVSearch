## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: pdf_faithful_mini_v15

## Fixed cross-backbone validation

- Runner: `pdf-faithful-v10-cross-node-localization`
- Git commit: `0fb3d4d`
- Code fingerprint in every result row: `1090faa3bb65b388e84a76255baf3fd7ae73597c79fa166774bfa7728db7f064`
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

The final answer-change gate additionally requires a minimum state-evidence floor, a worst-case three-paraphrase contrastive margin above 0.10, non-root localization for pure target-detail questions, and support from at least two distinct focus nodes for history rescues. Relation questions retain the whole-image overview even when their plan also contains auxiliary target-detail evidence.

| Backbone | Result rows | Tree nodes | Ranked candidates | State evaluations | Gap-parser fallbacks | Planner fallbacks | Actions observed |
|---|---:|---:|---:|---:|---:|---:|---|
| LLaVA-OV-7B | 12 | 892 | 186 | 90 | 75 | 5 | BACKTRACK, SPLIT, ZOOM |
| InternVL2.5-8B | 12 | 892 | 120 | 87 | 4 | 1 | BACKTRACK, EXPAND, NEXT, SPLIT, ZOOM |
| Qwen2.5-VL-7B | 12 | 763 | 84 | 83 | 0 | 6 | BACKTRACK, EXPAND, NEXT, SPLIT, ZOOM |

The LLaVA action-score JSON parser used its explicit analytic fallback in 75/90 states; this is visible rather than hidden. The uncertainty re-answering, ranking, tree traversal, and independent verifier remained active in those states. This parser robustness gap should be reported separately from answer accuracy.

## Validation boundary

This is a fixed small-scale engineering gate, not a paper-level full-benchmark estimate and not a statistical significance claim. The selected HR aggregates were previously exposed in this workspace. An exploratory v9 full-V* prefix exposed three InternVL regressions at ordinals 14, 16, and 17. They were retained as failure evidence and used to define the generic v10 cross-node, margin, and localization gates; v10 blocks all three while preserving the InternVL ordinal 116 and Qwen ordinal 132 corrections. Full v10 V* runs are reported separately when complete.

## Artifacts

- `reproduction/pdf_faithful/mini_v15/llava/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/mini_v15/internvl/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/mini_v15/qwen/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/smoke_v13/` contains the three full-prefix regression guards and the Qwen ordinal 19 guard.
- `reproduction/pdf_faithful/smoke_v14/` confirms that the final mixed relation/detail rule preserves the InternVL ordinal 116 and Qwen ordinal 132 corrections.
