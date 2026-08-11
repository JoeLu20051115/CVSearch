# Cross-Backbone Full Reproduction Design

## Goal

Measure the actual, locally reproduced gains of official CVSearch and the
current LOGIV_V2 selected method on LLaVA-OV-7B and InternVL2.5-8B.  The
experiment covers V* Bench, HR-Bench 4K, and HR-Bench 8K and does not use paper
scores as experimental controls.

## Comparisons

For each backbone and benchmark, produce three paired prediction vectors:

1. Direct answer using the official backbone wrapper.
2. Official CVSearch using the released backbone-specific search settings.
3. LOGIV_V2 using the same backbone and the frozen Phase-15 selection rules.

Report Attribute, Spatial, and Overall for V*; FSP, FCP, and Overall for each
HR-Bench resolution.  Recompute and report `CVSearch - Direct`,
`LOGIV_V2 - CVSearch`, and `LOGIV_V2 - Direct` from local predictions.  Paper
numbers are shown only as a reference and never substituted for missing local
results.

## Fixed Experimental Inputs

- Backbones: `lmms-lab/llava-onevision-qwen2-7b-ov` and
  `OpenGVLab/InternVL2_5-8B`.
- Benchmarks: the existing local V*, HR-Bench 4K, and HR-Bench 8K artifacts.
- Visual expert and NLP artifacts: the existing local SAM 3 and spaCy
  checkpoints.
- Evaluation: the repository's official benchmark evaluators, with output
  parsing checked against the trusted annotations.
- Decoding: deterministic decoding through the existing wrappers.
- LOGIV_V2 selection: the published Phase-15 B8 rule for V* and B7 semantic
  normalization for HR-Bench, without retuning thresholds on either backbone.

## Minimal Compatibility Work

Keep model-specific rendering and prompting inside the existing wrappers.  Add
only the common capabilities required by the LOGIV_V2 runners:

- dispatch the evidence-gap runtime to the existing LLaVA or InternVL wrapper;
- expose the already-computed per-option losses alongside the winning option;
- generate answer-free localization text through a wrapper-neutral method;
- record backbone-neutral artifact provenance while retaining the exact model
  identity.

Do not change the Qwen implementation, CVSearch search algorithm, benchmark
annotations, evaluator logic, LOGIV_V2 selector rules, or selection thresholds.

## Execution Flow

1. Create an isolated worktree and verify the existing unit-test baseline.
2. Add contract tests that fail for the missing LLaVA/InternVL capabilities,
   then implement the smallest compatibility changes that make them pass.
3. Acquire and fingerprint the two exact model snapshots.
4. Run one Direct, CVSearch, and LOGIV_V2 smoke record per backbone.
5. If every smoke run is structurally valid, run the full paired suite.  Use
   separate output directories per backbone and method and never overwrite the
   existing Qwen artifacts.
6. Run the official evaluators, independently recompute accuracy from trusted
   annotations, and verify row counts, unique identities, parseability, hashes,
   process exit codes, and absence of tracebacks/OOMs.
7. Publish a machine-readable score file and a concise comparison report.

## Parallelism and Recovery

Use separate GPUs for independent model/benchmark jobs only after each model's
smoke test succeeds.  Prediction files are resumable and remain model-scoped.
An individual failed job is reported with its command, log, and last completed
record; completed jobs are preserved and are not silently rerun or replaced.

## Acceptance Criteria

- All 18 backbone/benchmark/method score cells have locally generated complete
  prediction vectors, or any missing cell is explicitly classified as failed
  with retained diagnostic evidence.
- Every successful V* vector covers the evaluator's complete V* annotation set;
  every successful HR vector covers all 200 topics and 800 option cycles.
- Category and Overall scores are produced by the official evaluators and match
  an independent recomputation.
- The report distinguishes observed gains from paper-reported gains and makes no
  cross-backbone effectiveness claim unless both local paired comparisons are
  complete.

## Outputs

- Model-scoped raw predictions and logs under
  `reproduction/cross_backbone/`.
- `reproduction/cross_backbone/scores.json` containing counts, accuracies,
  deltas, hashes, and run status.
- `reproduction/cross_backbone/report.md` summarizing actual versus reported
  results and the cross-backbone verdict.
