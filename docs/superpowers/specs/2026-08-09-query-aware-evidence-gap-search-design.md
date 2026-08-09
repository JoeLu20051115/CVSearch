# Query-Aware Evidence-Gap Search Design

## Objective

Implement the method described in `Query感知证据缺口引导的自适应视觉搜索.pdf` as a training-free extension of the current CVSearch repository. The first experimental milestone keeps the exact reproduced `Qwen2.5-VL-7B-Instruct` checkpoint, SAM 3 checkpoint, image preprocessing, V* Bench / HR-Bench 4K / HR-Bench 8K annotations, and official evaluators fixed.

The primary acceptance target is to exceed the paired CVSearch results already reproduced in this workspace:

| Benchmark | Reproduced CVSearch | Initial target |
| --- | ---: | ---: |
| V* Bench | 87.43 | > 87.43 |
| HR-Bench 4K | 76.25 | > 76.25 |
| HR-Bench 8K | 75.12 | > 75.12 |

Accuracy must not be obtained by changing answer labels, evaluators, benchmark inputs, the base MLLM, or by tuning on the final evaluation partition. Search cost and termination type must be reported with accuracy.

## Method Interpretation

The proposal is not a new region generator. Its contribution is a stateful evidence-collection policy around the existing CVSearch candidate mechanism:

1. `q0` preserves the complete answer task.
2. `Qaug` adds only target-localization expressions.
3. `E(q0)` defines the evidence that a correct answer must visibly satisfy.
4. CVSearch/SAM 3 supplies expert regions and the semantic adaptive tree.
5. Query-aware ranking decides which region to inspect first but never certifies an answer.
6. Four non-exclusive evidence gaps decide the next action: `ZOOM`, `SPLIT`, `EXPAND`, or `NEXT`.
7. Answer stability and independent visual support decide whether to stop; they do not choose the action.
8. The controller keeps history, backtracks when progress stalls, and distinguishes `CERTIFIED_STOP` from `FORCED_RETURN`.

This separation is essential. Candidate priority, action gaps, answer uncertainty, and evidence support are different quantities and must remain separately observable in code and logs.

## Considered Approaches

### A. Full document-faithful execution from the first run

Run three answer queries and an independent verifier at every state, instantiate each action exactly, and enforce the complete certified-stop rule immediately.

This is the closest literal implementation, but it multiplies Qwen calls before any component has been validated. A bad score would be hard to attribute to ranking, routing, verifier calibration, or stopping.

### B. Feature-gated faithful architecture (selected)

Build all document concepts behind explicit modules and trace fields, but use an adaptive-cost default. Cheap ranking and analytic feasibility always run. Multi-query answering and independent verification run only when the current state is competitive for stopping or final selection. Every component can be disabled without changing the candidate pool or evaluator.

This approach supports a single end-to-end command while preserving the ability to isolate failures and ablate modules. It is the selected architecture.

### C. Query-aware reranking only

Apply CLIP main/augmented-query scores to the existing CVSearch candidates and retain the current answer path.

This is a useful ablation and smoke-test fallback, but it does not test the proposal's main claim that evidence gaps improve sequential control.

## Architecture

The new implementation is additive. Existing CVSearch files remain the baseline source of truth.

### Query planner

Input: original question and the existing CVSearch in-context examples.

Output: a `QueryPlan` containing:

- `main_query`: the unchanged original question;
- `targets`: CVSearch-compatible target expressions;
- `augmented_queries`: a small set of answer-free localization phrases;
- `evidence_items`: target, detail, relation/context, and optional coverage requirements;
- `global_scope_required`: true for count, absence, all, or uniqueness questions.

The Qwen2.5-VL-7B language base produces the plan once with deterministic decoding. A deterministic rule-based fallback is used only when structured parsing fails. Generated localization queries containing answer-option content are rejected.

### Candidate factory

Input: image, query targets, SAM 3 output, and its dense feature map.

Output: canonical `SearchCandidate` objects with original-image `xywh`, tree depth, parent/children identifiers, render level, source, CVSearch complexity prior, and canonical key.

The factory reuses `ConstrainedTreeBuilder` and `AdaptiveImageTree` with the paper parameters. Expert boxes and semantic-tree regions remain distinguishable. Deeper tree nodes are revealed to the policy through `SPLIT`; building the lightweight structural tree early is allowed, but unvisited nodes incur no MLLM observation cost.

Canonical keys contain rounded original coordinates, depth/scale, and render level. Duplicate observations are rejected.

### Query-aware ranker

For every sibling set, compute:

- main-query CLIP similarity;
- mean of the top three augmented-query similarities;
- CVSearch feature-complexity prior;
- image edge density.

Convert each component to a within-sibling percentile and freeze it. The default score is:

`relevance = beta * main + (1 - beta) * augmented`

`visual = lambda * complexity + (1 - lambda) * edge_density`

`rank = alpha * relevance + (1 - alpha) * visual`

All candidates remain available. Rank controls observation order only. The cached `openai/clip-vit-large-patch14-336` checkpoint is the default auxiliary model.

### Search state and actions

`SearchState` stores focus, context candidates, visited canonical keys, failed actions, remaining budget, global queue, history snapshots, current answer record, and progress measurements.

- `ZOOM`: keep the original coordinates and increase only rendered/token resolution.
- `SPLIT`: reveal and rank the current node's CVSearch children.
- `EXPAND`: add the highest-ranked unvisited spatial neighbor needed for a relation/context item while keeping the focus unchanged.
- `NEXT`: move focus to the next unvisited global candidate.
- `BACKTRACK`: restore the best historical state that still has an unvisited branch; consumed budget is not restored.

An action is feasible only when it can produce a new canonical observation inside the remaining budget. Failed no-op actions are recorded and cannot repeat unchanged.

### Evidence-gap scorer

The primary scorer asks the same Qwen2.5-VL-7B to emit four independent `[0,1]` gap values from `q0`, `E(q0)`, current observation, and compact history. Candidate answers and `Qaug` are excluded from this prompt.

A deterministic fallback derives gaps from target-existence confidence, crop scale, tree children, missing relation context, and queue state. The fallback is always logged. Feasibility masking occurs after scoring; the highest feasible gap selects one action.

### Answer stability

V* uses multiple semantically equivalent question prompts with the same option set. Per-option loss is averaged across prompts, and normalized top-two loss margin supplies the answer-confidence component.

HR-Bench already supplies four shuffled option lists for one semantic answer. Each generated letter is mapped back to its canonical option text, canonical answers are grouped, and the winning semantic answer is mapped back to the correct letter in every shuffle. This implements self-consistency without changing the official output schema.

Answer records retain all raw outputs, canonical groups, frequencies, margins, and final evaluator-compatible output.

### Independent support verifier

The initial verifier is the frozen CLIP checkpoint, which is independent of the Qwen answer generator. For each evidence item, it scores the current visual observation against an answer-conditioned evidence statement. Scores are calibrated within the state against neutral/negative prompts and quantized to the document's `0, 0.25, 0.5, 0.75, 1` scale.

Coverage evidence is computed mechanically from visited canonical keys, never by a model. The trace records mean and minimum support separately.

This is an explicit first-stage approximation to the document's independent multimodal verifier. A later cross-checkpoint MLLM verifier remains a feature-gated replacement, not a prerequisite for the first full benchmark.

### Stop controller

The controller preserves the document's ordering:

1. hard budget exhausted -> `FORCED_RETURN`;
2. all gap, uncertainty, average-support, and minimum-support gates pass -> `CERTIFIED_STOP`;
3. low actionable gap but failed confidence/support -> `BACKTRACK`;
4. otherwise execute the highest feasible gap action;
5. no-op -> try the next feasible action;
6. two steps with less than the configured progress delta -> `BACKTRACK`;
7. no history branch -> `NEXT`;
8. empty queue -> `FORCED_RETURN`.

Forced return selects the historical answer with the highest predeclared composite of answer stability, support minimum, support average, and cost. It is never reported as certified.

## Data Flow

1. Load the same Qwen, SAM 3, spaCy, annotations, and evaluator configuration as the reproduced CVSearch run.
2. Perform the existing global answer-sufficiency check.
3. Create `QueryPlan`; run SAM 3 once for proposals and features.
4. Build the CVSearch semantic tree and initial candidate queue.
5. Rank initial candidates with main/augmented CLIP and visual information.
6. Observe the best initial state; score gaps.
7. When a state is eligible for stopping or is a new historical best, produce the stability and support records.
8. Apply stop/backtrack/action control and repeat within the hard budget.
9. Return evaluator-compatible output plus a full trace in the JSONL record.
10. Score with the unmodified official scripts and summarize accuracy, cost, search modes, stop types, and module fallbacks.

## Isolation and Interfaces

New modules must not import the CLI or mutate global model state. The intended boundaries are:

- pure dataclasses and canonicalization;
- pure percentile/ranking utilities;
- CLIP scorer wrapper;
- query plan parser with deterministic fallback;
- candidate adapter around CVSearch tree objects;
- action controller with injected observation/gap/answer/support callbacks;
- benchmark-facing orchestration;
- separate CLI.

Pure control and parsing logic is unit-tested without GPUs. Model wrappers are exercised through one-sample integration tests.

## Failure Handling and Observability

Every sample records:

- query plan and whether fallback parsing was used;
- every candidate score component and frozen rank;
- every state, feasible mask, four raw gaps, chosen action, and no-op reason;
- raw and grouped answers, uncertainty, per-evidence support, and stop-gate result;
- budget/cost counters and termination type;
- final searched boxes in original coordinates.

Structured-generation failure falls back per sample and does not crash the benchmark. CUDA OOM, missing artifacts, invalid image paths, or corrupted output remain hard failures and are not silently retried.

## Experimental Protocol

### Stage 0: baseline lock

Re-run static parameter tests and evaluators on the retained six answer files. Record the exact checkpoint revision and preprocessing values.

### Stage 1: CPU unit tests

Test canonical keys, percentile ties, no-pruning ranking, option canonicalization, feasibility masks, no-op prevention, progress/backtrack rules, certified versus forced termination, and trace serialization.

### Stage 2: one-sample integration

Run one V* and one HR sample through Direct, retained CVSearch, rerank-only, and the full default controller. Validate output schema and evaluator compatibility.

### Stage 3: label-blind smoke set

Use a fixed, source-grouped sample list stratified by benchmark and retained CVSearch search mode. Inspect crashes, fallbacks, cost, and action diversity without using answer correctness to tune.

### Stage 4: development calibration

Create a deterministic source-grouped development partition before examining new-method results. Tune only declared thresholds and fusion weights on that partition. Prefer a broad coarse grid followed by single-factor refinement. Do not tune on the final partition.

### Stage 5: frozen evaluation

Freeze configuration and run all three final partitions in parallel, one H200 per benchmark. Then run the official evaluators once. If the target is not met, use traces to identify one failing module, revise on development data, and perform a new frozen evaluation with a new configuration identifier.

## Ablations

The same candidate pool and budget support:

- CVSearch order / CLIP only / visual only / combined;
- main only / augmented only / main plus top three;
- fixed action / direct four-score query / analytic fallback / hybrid controller;
- no re-scoring / no backtracking;
- raw answer / semantic self-consistency / self-consistency plus support;
- average support only / average plus minimum;
- certified-stop disabled / full controller.

## Success Criteria

The first engineering gate requires:

- all CPU tests and two GPU integration samples pass;
- no candidate is removed solely by CLIP score;
- every action changes the canonical observation or is logged as a no-op and suppressed;
- exact benchmark output counts and official evaluator compatibility;
- separate certified-stop and forced-return metrics;
- no untracked fallback or malformed trace.

The research gate requires frozen-partition improvement over paired CVSearch at equal declared cost, or equal accuracy with a measurable reduction in MLLM calls/processed pixels. Full-benchmark numbers are reported only after the configuration is frozen.

## Non-goals for the First Iteration

- training or fine-tuning any model;
- replacing Qwen2.5-VL-7B with a stronger generator;
- changing CVSearch's SAM/tree parameters;
- introducing a new region proposal algorithm;
- using benchmark labels inside the search policy;
- claiming statistical certification from the rule-based `CERTIFIED_STOP` name.
