# Stage 3b SPLIT Calibration and Ordering Rescue Design

## Decision status

Approved by the user's 2026-08-13 instruction to retain commit `32f395d`,
repair only Qwen SPLIT calibration and V*/TreeBench ordering on development
data, freeze the policy, run a new unseen two-backbone/four-dataset gate, and
start the full benchmark immediately if all promotion criteria pass.

## Objective and frozen boundary

Stage 3b must preserve the Stage 3 v3 result at `32f395d`: +7 official units
relative to Stage 2 with zero corruptions. It is a fail-closed rescue layer,
not a replacement policy.

For every row, replay the existing Stage 3 v3 selector first on the original
four branches and with the original calibration manifest. If v3 selects a
SPLIT answer, return that exact output and do not let Stage 3b inspect or
override it. Stage 3b is eligible only when v3 returns the exact Stage 2/P0
output. Any identity, provenance, calibration, schema, or budget failure also
returns that exact output.

The committed source tree and compact result artifacts at `32f395d` are the
rollback point. Existing raw runs are read-only inputs and remain untracked.

## Diagnosis from development data

Two independent failures are visible on already-opened development rows:

1. The Qwen Stage 2 isotonic support mapping is not valid for SPLIT views. On
   V*/TreeBench geometric-sufficiency labels it changes Brier from 0.1197 to
   0.3350 and ECE from 0.0560 to 0.3827 while leaving AUROC unchanged. The
   ranking signal is useful, but the probability mapping is shifted.
2. The current tree expands only the top two of four depth-one roots. GT boxes
   used offline on development show root Recall@2 of 16/20 on V* and 11/12 on
   TreeBench, while expanding all four roots has complete root coverage by
   construction. This is a candidate-ordering ceiling rather than a selector
   threshold problem.

Historical dense 2x2+3x3+4x4 search is not revived. It generated 29 candidates
per row and produced large always-replace regressions. Stage 3b keeps the PDF
main line: a bounded coarse-to-fine 2x2 tree with query-guided ordering,
support screening, and backtracking.

## SPLIT-specific calibration

Fit and freeze a separate support calibration for SPLIT observations using
only previously opened development rows. Labels are geometric evidence
sufficiency derived from GT boxes for V* and TreeBench; GT boxes are never
serialized into inference inputs or replay records.

Candidate calibration families are exact identity and deterministic isotonic.
Selection is source-grouped cross-validation, lexicographically minimizing
Brier, then ECE, then preferring identity on a tie, subject to AUROC not
worsening. Calibration is selected independently per backbone because raw
support distributions are model-specific; action thresholds and search logic
remain shared. HR rows receive the frozen backbone mapping but do not
contribute labels to fitting.

The v3 prefix continues to use its original Stage 2 calibration. Only Stage 3b
rescue branches use the new SPLIT calibration, so calibration repair cannot
change any of the locked +7 decisions.

## Coverage-preserving tree order

The first eight depth-two support probes and first four answer-bearing branches
are byte-for-byte the existing v3 prefix: expand the top two CLIP-ranked roots,
screen their eight leaves, then answer the two CLIP seeds and two strongest
remaining support candidates.

Only after the prefix is complete, screen the four leaves under each of the two
previously omitted roots. Append at most one rescue branch per omitted root,
choosing its highest shared CLIP/support-ranked leaf. This yields fixed bounds
of sixteen answer-free depth-two probes and six answer-bearing branches. Root
and leaf tie-breaking remains stable path order.

Each rescue branch uses tight, medium-context, and context views. A changed
answer requires agreement from at least two distinct render hashes, sufficient
raw and calibrated support, and no equally strong P0 conflict. This third view
addresses development cases where the tight crop is correct but the old
context view is not, without weakening the two-view confirmation rule.

Stage 3b examines rescue branches only after v3 abstains. It cannot route on
backbone name, dataset name, resolution, ordinal, label, correctness, GT box,
or evaluator category. Backbone identity is used only to load the frozen
support mapping, matching the existing calibration contract.

## Development and promotion protocol

All calibration and search-policy work is restricted to already opened Stage
3 development data. GT boxes may be used there only for Recall@K, first-evidence
rank, and geometric-sufficiency calibration labels.

Before any new model inference, create and commit a deterministic
`validation_v3` manifest excluding every ordinal in the Stage 1/2/3, v1, and
v2 manifests. The exact remaining capacities are 21 V*, 152 HR-Bench 4K, 152
HR-Bench 8K, and 349 TreeBench topics. Select 20 V* and 12 topics from each
other dataset using SHA-256 ordering over dataset, ordinal, and input image.
The selection procedure cannot read answers, correctness, or boxes.

After code, calibration, policy, artifact hashes, and decision hashes are
frozen, run `validation_v3` once for both backbones and all four datasets. Do
not tune after opening its labels. Promote directly to the complete benchmark
only if:

- all eight backbone/dataset cells have zero negative delta;
- pooled Qwen and pooled InternVL deltas are both positive;
- pooled V*, HR-Bench 4K, HR-Bench 8K, and TreeBench deltas are all positive;
- corrections exceed corruptions and aggregate accuracy increases;
- calibration, provenance, budget, JSONL, and operational audits pass.

If any condition fails, stop before full inference and report the exact failed
cells. Validation results may reject Stage 3b but may not be used for another
round of tuning.

## Operational policy

GPU work starts only after CPU tests and frozen manifests pass. Do not kill or
reuse unrelated GPU processes. Jobs use isolated `adaptive_search_v7` raw and
report directories and pinned runner/config/calibration hashes. Full inference
starts automatically after the frozen gate passes, using the number of GPUs
that are safely available at launch time.
