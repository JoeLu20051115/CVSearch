## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-12
- Verification Status: VERIFIED
- Version Label: pdf_faithful_mini_v12

## Fixed cross-backbone validation

- Runner: `pdf-faithful-v9-state-evidence-floor`
- Git commit: `7699d53`
- Code fingerprint in every result row: `77fc119be177f77f56431a96910fd81af52dc168d5d489367fbcb52b15a29e44`
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

| Backbone | Result rows | Tree nodes | Ranked candidates | State evaluations | Gap-parser fallbacks | Planner fallbacks | Actions observed |
|---|---:|---:|---:|---:|---:|---:|---|
| LLaVA-OV-7B | 12 | 892 | 186 | 90 | 75 | 5 | BACKTRACK, SPLIT, ZOOM |
| InternVL2.5-8B | 12 | 892 | 120 | 87 | 4 | 1 | BACKTRACK, EXPAND, NEXT, SPLIT, ZOOM |
| Qwen2.5-VL-7B | 12 | 763 | 84 | 83 | 0 | 6 | BACKTRACK, EXPAND, NEXT, SPLIT, ZOOM |

The LLaVA action-score JSON parser used its explicit analytic fallback in 75/90 states; this is visible rather than hidden. The uncertainty re-answering, ranking, tree traversal, and independent verifier remained active in those states. This parser robustness gap should be reported separately from answer accuracy.

## Validation boundary

This is a fixed small-scale engineering gate, not a paper-level full-benchmark estimate and not a statistical significance claim. The selected HR aggregates were previously exposed in this workspace. Full V* runs for all three backbones were launched only after each backbone passed this gate; their results must be reported separately when complete.

## Artifacts

- `reproduction/pdf_faithful/mini_v12/llava/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/mini_v12/internvl/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/mini_v12/qwen/{vstar,hr4,hr8}.jsonl`
- `reproduction/pdf_faithful/smoke_v12/` contains the targeted guard checks that blocked Qwen V* ordinal 19 while preserving the Qwen ordinal 132 and InternVL ordinal 116 corrections.
