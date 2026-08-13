# Fixed-Pool SPLIT Ranking Ablation Design

## Decision status

Approved by the user's 2026-08-13 instruction to implement a strict ranking
ablation and failure decomposition over the fixed sixteen-patch Stage 3b pool,
and to continue without intermediate approval until the work succeeds.

## Objective

Establish whether the Stage 3b query-aware ranking places evaluator-defined
visual evidence earlier than non-query and single-signal baselines when every
method receives the same sixteen depth-two patches and the same K. Separately,
explain why frozen Stage 3b candidate opportunities did or did not become final
answer corrections.

This is an evaluator-only diagnostic. It does not change inference, candidate
generation, the selector, or any frozen validation decision.

## Scope and leakage boundary

Ranking uses only the already-opened V* and TreeBench development partitions,
because those are the two datasets with evaluator geometry. The two backbone
copies are required to have identical question, image, and sixteen-candidate
`(path, box)` sets. Their score order is evaluated separately because each
backbone produces its own answer-free query plan; score identity is neither
expected nor required. Reports expose both 32 unique source topics and 64
topic/backbone ranking evaluations, with per-backbone results.

Failure decomposition uses the already-frozen `validation_v3` report and its
raw observations as a post-hoc explanation across both backbones and all four
datasets. Validation labels may populate diagnostic counts but may not select
weights, thresholds, policies, or code paths.

## Considered approaches

1. Reconstruct ranking components from frozen logs and pixels. Recompute the
   deterministic edge/deviation visual term, then recover the CLIP relevance
   percentile from the frozen equation
   `combined = 0.7 * clip + 0.3 * visual`. This is the selected approach: it is
   exact, CPU-only, and cannot introduce a new CLIP preprocessing run.
2. Rerun CLIP over all sixteen crops. This would expose raw globally comparable
   similarities, but adds GPU/model provenance and risks measuring a different
   preprocessing stack from the frozen run.
3. Compare only the six recorded visit positions. This is cheap but cannot
   separate CLIP from visual information and changes the candidate pool across
   ranking policies.

## Ranking contract

Every eligible topic must expose exactly four roots, four unique children per
root, and sixteen unique depth-two support probes. The fixed leaf identity is
`(path, XYXY box)`.

Visual and recovered CLIP components are computed independently within the
four root siblings and within each four-child sibling set, matching the
runtime percentile contract. A leaf receives the unweighted mean of its root
and child component. This combines the two fixed tree levels without tuning a
new coefficient. Ties are resolved by path.

The five policies are:

- `random_expected`: exact combinatorial expectation, not sampled seeds;
- `grid`: ascending two-index path;
- `visual_only`: root/child edge-plus-deviation information;
- `clip_only`: recovered root/child CLIP relevance percentile;
- `combined`: the frozen 0.7 CLIP plus 0.3 visual score.

For K in `(1, 3, 6)`, a topic is a hit when at least one of the first K leaf
crops satisfies the existing V* or TreeBench evaluator geometry proxy. Report
topic Recall@K, pool recall, MRR, mean first-evidence rank over pool hits, and
the complete first-rank distribution. Also report each dataset separately.

## Failure decomposition contract

For each official unit in the frozen validation report:

- a correct Stage 2 unit that becomes wrong is `corruption`;
- an incorrect Stage 2 unit corrected by Stage 3b is `converted`;
- an incorrect unit with a correct observed candidate but unchanged output is
  `selector_abstained`;
- an incorrect unit with a correct observed candidate but a different wrong
  selected output is `selector_wrong_choice`;
- an incorrect unit with no correct observed candidate is
  `no_correct_observed_answer`.

For V*/TreeBench, the last category is refined in order into
`geometry_pool_miss`, `six_branch_budget_miss`, or `vlm_answer_miss` using the
same evaluator-only geometry proxy. HR units remain in the answer-level class
because no target geometry is available. Counts must exactly partition all
Stage 2 errors, and corruption counts must reproduce the frozen report.

## Success gates

The evaluator succeeds only when all structural, hash, and accounting audits
pass. The current combined ranking is considered empirically successful when:

- combined Recall@3 is strictly above exact random expectation and grid;
- combined Recall@3 is at least both single-signal policies;
- combined Recall@6 and MRR are at least both single-signal policies;
- combined Recall@3 is no lower than grid for either V* or TreeBench;
- all failure categories exactly reconcile with frozen aggregate counts.

If a metric gate fails, the report must say so. No validation-driven tuning is
allowed. A later ranking redesign may use development-only grouped selection
and must receive a new unseen evaluation source.

## Artifacts

- `cvsearch/eval/eval_split_ranking_ablation.py`: pure evaluator and CLI.
- `tests/test_eval_split_ranking_ablation.py`: structural, metric, leakage,
  accounting, and CLI tests.
- `reproduction/evidence_gap/reports/split-ranking-ablation-v1.json`: compact
  reproducible result with input hashes and explicit gates.
