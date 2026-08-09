# Query-Aware Evidence-Gap Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a leak-free, resumable Query-aware evidence-gap extension that keeps Qwen2.5-VL-7B and CVSearch's candidate generator fixed while improving the strongest retained CVSearch baseline on V*, HR-Bench 4K, and HR-Bench 8K.

**Architecture:** Add pure policy/ranking/answer utilities and inject them through optional hooks into CVSearch so the default baseline remains byte-for-byte behaviorally unchanged. The first runnable milestone performs no-pruning Query-aware reranking, semantic answer aggregation, lightweight STOP/NEXT historical selection, full tracing, and atomic checkpointing; higher-cost actions and verifier calls are enabled only after the minimal loop passes development evaluation.

**Tech Stack:** Python 3.11, `unittest`, PyTorch 2.7.1, Transformers 4.57.0, Qwen2.5-VL-7B snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5`, SAM 3, spaCy 3.8.7, CLIP-L/14 snapshot `32bd64288804d66eefd0ccbe215aa642df71cc41`, NVIDIA H200 NVL.

## Global Constraints

- Policy code receives only `question`, `options`, `answer_type`, and `input_image`; benchmark truth and metadata never enter planning, ranking, routing, support, or stopping.
- Preserve CVSearch's SAM/tree parameters and existing behavior when every new hook is `None`.
- CLIP reorders candidates but never removes one.
- Keep both Qwen quick gates (`0.6` upstream and `0.8` paper-aligned); beat the per-benchmark envelope `87.43 / 76.625 / 76.75`.
- Tune on a deterministic source-grouped development partition only; freeze configuration before holdout/full evaluation.
- Every sample records raw score components, action decisions, fallbacks, MLLM calls, processed pixels, elapsed time, and termination type.
- `CERTIFIED_STOP` and `FORCED_RETURN` are distinct. A failed parser/verifier or exhausted budget cannot become certified.
- Crashes are diagnosed from complete logs and are never silently retried; resume continues from validated partial records.
- Existing uncommitted user changes and retained reproduction outputs are preserved.

---

## File Map

- `cvsearch/evidence_gap/types.py`: immutable policy-facing types and serialization.
- `cvsearch/evidence_gap/input.py`: annotation whitelist and deterministic split membership.
- `cvsearch/evidence_gap/baselines.py`: gate reconstruction and baseline envelope.
- `cvsearch/evidence_gap/answers.py`: HR canonical answer grouping and V* loss aggregation.
- `cvsearch/evidence_gap/ranking.py`: percentile transforms, visual statistics, and score fusion.
- `cvsearch/evidence_gap/clip_scorer.py`: frozen Transformers CLIP wrapper.
- `cvsearch/evidence_gap/policy.py`: feasibility, STOP/NEXT selection, history, budget, and termination.
- `cvsearch/evidence_gap/io.py`: resumable per-sample JSONL checkpoint writer.
- `cvsearch/evidence_gap/method.py`: Query plan construction and hook composition.
- `cvsearch/perform_EGSearch.py`: sanitized benchmark CLI.
- `cvsearch/run_eval_evidence_gap.sh`: strict one-command launcher.
- `cvsearch/CVSearch.py`: optional node-ranker/answer-selection hooks only.
- `cvsearch/models/modeling_qwenvl.py`: option losses and structured answer details.
- `tests/test_evidence_gap_*.py`: CPU regression tests.
- `reproduction/evidence_gap/`: ignored raw partials/logs; tracked scores and report.

### Task 1: Lock leak-free inputs and the dual baseline envelope

**Files:**
- Create: `cvsearch/evidence_gap/__init__.py`
- Create: `cvsearch/evidence_gap/input.py`
- Create: `cvsearch/evidence_gap/baselines.py`
- Create: `tests/test_evidence_gap_input.py`
- Create: `tests/test_evidence_gap_baselines.py`

**Interfaces:**
- Produces: `sanitize_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]`.
- Produces: `split_bucket(benchmark: str, input_image: str, seed: int = 260809) -> Literal["dev", "holdout"]`.
- Produces: `reconstruct_quick_gate(direct_rows, search_rows, threshold) -> list[dict]`.
- Produces: `score_rows(benchmark, rows) -> float` and `baseline_envelope(...) -> dict[str, float]`.

- [ ] **Step 1: Write failing annotation-whitelist tests**

```python
class AnnotationSanitizationTest(unittest.TestCase):
    def test_vstar_truth_is_not_policy_visible(self):
        raw = {"question": "q", "options": ["a"], "answer_type": "logits_match",
               "input_image": "i.jpg", "bbox": [[1, 2, 3, 4]], "target_object": ["comb"]}
        self.assertEqual(sanitize_annotation(raw), {
            "question": "q", "options": ["a"], "answer_type": "logits_match", "input_image": "i.jpg"})

    def test_hr_truth_and_metadata_are_not_policy_visible(self):
        raw = {"question": "q", "options": ["A. x"], "answer_type": "option_list",
               "input_image": "0.jpg", "answer": ["A"], "category": "single", "index": 0}
        self.assertEqual(set(sanitize_annotation(raw)), {"question", "options", "answer_type", "input_image"})
```

- [ ] **Step 2: Run the whitelist tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_input -v`

Expected: import failure for `cvsearch.evidence_gap.input`.

- [ ] **Step 3: Implement the whitelist and source-grouped split**

```python
POLICY_FIELDS = ("question", "options", "answer_type", "input_image")

def sanitize_annotation(annotation):
    missing = [key for key in POLICY_FIELDS if key not in annotation]
    if missing:
        raise ValueError(f"missing policy fields: {missing}")
    return {key: annotation[key] for key in POLICY_FIELDS}

def split_bucket(benchmark, input_image, seed=260809):
    digest = hashlib.sha256(f"{seed}:{benchmark}:{input_image}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") % 5 == 0 else "holdout"
```

- [ ] **Step 4: Write and run baseline reconstruction tests**

Test strict `root_ans_conf > threshold`, record copying, V* `output == 0`, and HR letter parsing exactly as the official evaluators. Run:

`PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_baselines -v`

Expected after implementation: PASS and reconstructed retained scores round to `86.91`, `76.625`, and `76.75` for gate 0.6.

- [ ] **Step 5: Run existing parameter tests and commit**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_paper_parameters tests.test_evidence_gap_input tests.test_evidence_gap_baselines -v`

Expected: all tests pass.

```bash
git add cvsearch/evidence_gap/__init__.py cvsearch/evidence_gap/input.py cvsearch/evidence_gap/baselines.py tests/test_evidence_gap_input.py tests/test_evidence_gap_baselines.py
git commit -m "test: lock evidence-search evaluation gates"
```

### Task 2: Define canonical state, score, budget, and trace types

**Files:**
- Create: `cvsearch/evidence_gap/types.py`
- Create: `tests/test_evidence_gap_types.py`

**Interfaces:**
- Produces: `canonical_key(bbox, depth, render_level) -> str`.
- Produces dataclasses: `QueryPlan`, `CandidateScore`, `SearchCandidate`, `BudgetLedger`, `AnswerRecord`, `HistoryRecord`, `StepTrace`, and `MethodTrace`.
- `BudgetLedger.consume(kind: str, amount: int | float) -> None` raises `BudgetExceeded` before an over-budget operation.

- [ ] **Step 1: Write failing canonicalization and budget tests**

```python
class EvidenceGapTypesTest(unittest.TestCase):
    def test_canonical_key_is_stable_for_numeric_bbox_types(self):
        self.assertEqual(canonical_key([1, 2, 30, 40], 2, 1),
                         canonical_key([1.0, 2.0, 30.0, 40.0], 2, 1))

    def test_budget_rejects_before_mutating(self):
        budget = BudgetLedger(max_mllm_calls=2, max_processed_pixels=100)
        budget.consume("mllm_calls", 2)
        with self.assertRaises(BudgetExceeded):
            budget.consume("mllm_calls", 1)
        self.assertEqual(budget.mllm_calls, 2)
```

- [ ] **Step 2: Run tests and verify missing symbols**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_types -v`

Expected: import failure for the new types.

- [ ] **Step 3: Implement focused dataclasses and JSON-safe `to_dict` methods**

Use string action values `ZOOM`, `SPLIT`, `EXPAND`, `NEXT`, `BACKTRACK`, `CERTIFIED_STOP`, and `FORCED_RETURN`. `SearchCandidate` stores `node`, but `to_dict()` serializes only identifiers, bbox, parent key, depth, source, render level, and `CandidateScore`.

```python
class BudgetExceeded(RuntimeError):
    pass

@dataclass
class BudgetLedger:
    max_mllm_calls: int
    max_processed_pixels: int
    mllm_calls: int = 0
    processed_pixels: int = 0

    def consume(self, kind, amount):
        current = getattr(self, kind)
        limit = getattr(self, f"max_{kind}")
        if current + amount > limit:
            raise BudgetExceeded(f"{kind}: {current}+{amount}>{limit}")
        setattr(self, kind, current + amount)

def canonical_key(bbox, depth, render_level):
    coords = tuple(int(round(float(value))) for value in bbox)
    return f"{coords[0]}:{coords[1]}:{coords[2]}:{coords[3]}:d{depth}:r{render_level}"
```

- [ ] **Step 4: Verify round-trip serialization and commit**

Run the type tests plus `python -m json.tool` on a serialized fixture created by the test. Expected: PASS and valid JSON.

```bash
git add cvsearch/evidence_gap/types.py tests/test_evidence_gap_types.py
git commit -m "feat: add evidence-search state contracts"
```

### Task 3: Add evaluator-compatible semantic answer aggregation

**Files:**
- Create: `cvsearch/evidence_gap/answers.py`
- Create: `tests/test_evidence_gap_answers.py`
- Modify: `cvsearch/models/modeling_qwenvl.py:385-440`

**Interfaces:**
- Produces: `parse_option_block(block: str) -> dict[str, str]`.
- Produces: `aggregate_hr_answers(option_blocks: list[str], raw_outputs: list[str]) -> AnswerRecord`.
- Produces: `aggregate_vstar_losses(loss_rows: list[list[float]]) -> AnswerRecord`.
- Adds: `ModelQwenVL.multiple_choices_with_losses(...) -> tuple[int, list[float]]`; existing `multiple_choices_inference` delegates and returns only the integer.

- [ ] **Step 1: Write failing HR canonicalization tests**

```python
def test_shuffled_letters_vote_for_one_semantic_answer(self):
    blocks = ["A. red\nB. blue\n", "A. blue\nB. red\n", "A. red\nB. blue\n"]
    record = aggregate_hr_answers(blocks, ["A", "B.", "A"])
    self.assertEqual(record.canonical_answer, "red")
    self.assertEqual(record.output, ["A", "B", "A"])
    self.assertEqual(record.frequency, 1.0)
```

- [ ] **Step 2: Write failing V* loss aggregation tests**

Verify elementwise mean loss, minimum-loss option, top-two positive margin, and deterministic lower-index tie breaking.

- [ ] **Step 3: Run tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_answers -v`

Expected: missing implementation.

- [ ] **Step 4: Implement pure aggregators, then minimally refactor Qwen option loss**

Move only the final `loss_list` return boundary: stack detached float losses, return `(argmin, losses)`, and keep the existing public method's integer output unchanged.

```python
def multiple_choices_inference(self, image_pil, question, options, searched_nodes=None):
    choice, _ = self.multiple_choices_with_losses(image_pil, question, options, searched_nodes)
    return choice

# At the end of multiple_choices_with_losses:
loss_values = [float(loss.detach().cpu()) for loss in loss_list]
return int(torch.tensor(loss_values).argmin().item()), loss_values
```

- [ ] **Step 5: Run unit and baseline regression tests and commit**

Run all `tests.test_evidence_gap_answers` tests and the retained one-sample V* direct smoke. Expected: the original direct option remains unchanged and loss details are finite.

```bash
git add cvsearch/evidence_gap/answers.py cvsearch/models/modeling_qwenvl.py tests/test_evidence_gap_answers.py
git commit -m "feat: add semantic answer stability scores"
```

### Task 4: Implement no-pruning Query-aware ranking

**Files:**
- Create: `cvsearch/evidence_gap/ranking.py`
- Create: `cvsearch/evidence_gap/clip_scorer.py`
- Create: `tests/test_evidence_gap_ranking.py`

**Interfaces:**
- Produces: `percentile(values: Sequence[float]) -> list[float]` with average ranks for ties.
- Produces: `edge_density(image: PIL.Image.Image) -> float`.
- Produces: `fuse_scores(main, augmented, complexity, edge, beta, alpha, visual_lambda) -> list[CandidateScore]`.
- Produces: `ClipScorer.score(images: list[Image.Image], texts: list[str]) -> list[list[float]]` using normalized image/text embeddings and local-files-only loading.
- Produces: `QueryAwareNodeRanker.rank(nodes, image_pil, main_query, augmented_queries) -> list`.

- [ ] **Step 1: Write failing percentile/no-pruning tests**

```python
def test_all_candidates_survive_and_ties_are_equal(self):
    nodes = [FakeNode("a"), FakeNode("b"), FakeNode("c")]
    ranked, details = ranker.rank_with_details(nodes, image, "bus sign", ["blue road sign"])
    self.assertCountEqual([n.id for n in ranked], ["a", "b", "c"])
    self.assertEqual(len(details), 3)
```

Also test constant lists, one-element lists, score bounds, and input-list non-mutation.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_ranking -v`

- [ ] **Step 3: Implement pure ranking and edge density**

Compute edge density with NumPy finite differences on grayscale pixels; do not add OpenCV. Validate every weight is in `[0,1]` and every component length matches.

- [ ] **Step 4: Implement the frozen CLIP wrapper**

Load `/mnt/data3/data_xingrui/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41` with `local_files_only=True`, `eval()`, and inference mode. Unit tests inject a fake scorer; no unit test loads weights.

```python
self.processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
self.model = CLIPModel.from_pretrained(path, local_files_only=True).to(device).eval()

with torch.inference_mode():
    batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.device)
    outputs = self.model(**batch)
    image_features = torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
    text_features = torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
    return (image_features @ text_features.T).float().cpu().tolist()
```

- [ ] **Step 5: Run a one-batch CLIP integration probe and commit**

Run one real image against `question`, `target phrase`, and an unrelated phrase. Expected: finite matrix of shape `1x3`, peak GPU allocation below the H200 limit, and no network access.

```bash
git add cvsearch/evidence_gap/ranking.py cvsearch/evidence_gap/clip_scorer.py tests/test_evidence_gap_ranking.py
git commit -m "feat: add query-aware candidate ranking"
```

### Task 5: Add optional CVSearch hooks without changing baseline behavior

**Files:**
- Modify: `cvsearch/CVSearch.py:15-478`
- Modify: `cvsearch/CVSearch.py:569-772`
- Create: `tests/test_evidence_gap_hooks.py`

**Interfaces:**
- Add optional `node_ranker=None`, `answer_observer=None`, and `method_trace=None` parameters to `get_cvsearch_response`.
- Add optional `node_ranker=None` and `rank_context=None` to `semantic_guide_search_dynamic_depth`.
- `node_ranker(nodes, image_pil, main_query, augmented_queries) -> tuple[list, list[dict]]`.
- `answer_observer(observation_name, searched_nodes, raw_answer) -> None`.

- [ ] **Step 1: Write an AST/default-hook regression test**

The test verifies all hooks default to `None`, every existing caller remains valid, and the no-hook queue calls the original `calc_score_and_sort` result without extra filtering.

- [ ] **Step 2: Write a fake-ranker ordering test**

Create fake nodes and a fake ranker that reverses them. Assert the search visits reversed order and that the ranker receives the full candidate count.

- [ ] **Step 3: Run tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_hooks -v`

- [ ] **Step 4: Implement the smallest hook injection**

Apply the ranker after CVSearch computes valid nodes and before `execute_stage_search`. Never call the hook on pruned `is_evaluated=False` nodes because those are already excluded by the baseline's visual-complexity rule; record this candidate-pool boundary explicitly.

```python
Q = calc_score_and_sort(nodes_by_depth[depth], use_child_info=current_use_child_info)
if node_ranker is not None and Q:
    Q, rank_details = node_ranker(
        Q, image_pil, question, rank_context.get("augmented_queries", [visual_cue]))
    if method_trace is not None:
        method_trace.candidate_ranks.extend(rank_details)
```

- [ ] **Step 5: Run paper-parameter, hook, and retained evaluator parity tests**

Expected: all tests pass; scoring the retained answer files remains exactly `87.43 / 76.25 / 75.12`.

```bash
git add cvsearch/CVSearch.py tests/test_evidence_gap_hooks.py
git commit -m "feat: expose non-breaking CVSearch policy hooks"
```

### Task 6: Implement lightweight evidence gate, history, and answer selection

**Files:**
- Create: `cvsearch/evidence_gap/policy.py`
- Create: `tests/test_evidence_gap_policy.py`

**Interfaces:**
- Produces: `feasible_actions(state) -> tuple[str, ...]`.
- Produces: `should_certify(gaps, answer, support_avg, support_min, thresholds) -> bool`.
- Produces: `HistoryBuffer.add(record)`, `best()`, and `best_with_unvisited_branch()`.
- Produces: `select_root_or_search(root: AnswerRecord, search: AnswerRecord, tolerance: float) -> AnswerRecord`.

- [ ] **Step 1: Write failing stop/no-op/history tests**

Test all four stop gates, empty feasible-set convention, forced return on budget, no-op suppression, two-step `<0.03` progress backtrack, and deterministic history tie breaking by support minimum, support average, lower cost, then earlier step.

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_policy -v`

- [ ] **Step 3: Implement pure controller logic**

The minimal runtime enables `NEXT`, historical root/search selection, and termination. `ZOOM`, `SPLIT`, and `EXPAND` feasibility functions exist but are feature-disabled until their observation adapters pass integration tests.

```python
def should_certify(gaps, answer, support_avg, support_min, thresholds):
    return (
        max(gaps.values(), default=0.0) < thresholds.gap
        and answer.uncertainty < thresholds.uncertainty
        and support_avg >= thresholds.support_avg
        and support_min >= thresholds.support_min
    )

def select_root_or_search(root, search, tolerance):
    if search.confidence + tolerance < root.confidence:
        return dataclasses.replace(root, selected_from="root")
    return dataclasses.replace(search, selected_from="search")
```

- [ ] **Step 4: Run tests and commit**

```bash
git add cvsearch/evidence_gap/policy.py tests/test_evidence_gap_policy.py
git commit -m "feat: add evidence-gated answer fallback"
```

### Task 7: Build the sanitized method and resumable CLI

**Files:**
- Create: `cvsearch/evidence_gap/io.py`
- Create: `cvsearch/evidence_gap/method.py`
- Create: `cvsearch/perform_EGSearch.py`
- Create: `cvsearch/run_eval_evidence_gap.sh`
- Create: `tests/test_evidence_gap_io.py`
- Create: `tests/test_evidence_gap_method.py`

**Interfaces:**
- Produces: `JsonlCheckpointWriter(final_path, expected_ordinals)` using `<final>.partial`, per-record flush/fsync, duplicate rejection, and atomic `os.replace` on finalize.
- Produces: `build_query_plan(policy_annotation, targets) -> QueryPlan` with answer-free deterministic augmented phrases.
- Produces: `get_evidence_gap_response(..., policy_annotation, original_annotation, config) -> tuple[Any, MethodTrace]`.
- CLI adds `--ordinals`, `--split {all,dev,holdout}`, `--config`, `--resume`, and all existing model/data path arguments.

- [ ] **Step 1: Write crash/resume writer tests**

Write two records, reopen with resume, reject a duplicate ordinal, append the third, finalize, and assert the final file contains exactly three ordered JSON objects while `.partial` is absent.

- [ ] **Step 2: Write method leak-canary tests**

Pass truth fields whose values raise on access; fake planner/ranker/policy callbacks must complete using only the sanitized mapping. Assert the final emitted record restores evaluator fields only after the policy response is complete.

- [ ] **Step 3: Run tests and verify failure**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest tests.test_evidence_gap_io tests.test_evidence_gap_method -v`

- [ ] **Step 4: Implement IO and method composition**

Default config `minimal_v1` pins `beta=0.6`, `alpha=0.65`, `visual_lambda=0.5`, quick gate `0.8`, root fallback tolerance `0.05`, and a hard per-episode MLLM/pixel budget recorded in the output. These are starting development values, not final benchmark-tuned values.

```python
class JsonlCheckpointWriter:
    def write(self, ordinal, record):
        if ordinal in self.completed:
            raise ValueError(f"duplicate ordinal: {ordinal}")
        payload = dict(record, _eg_ordinal=ordinal)
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.completed.add(ordinal)

    def finalize(self):
        if self.completed != self.expected_ordinals:
            raise ValueError("cannot finalize incomplete output")
        self.handle.close()
        os.replace(self.partial_path, self.final_path)
```

- [ ] **Step 5: Implement strict launcher**

Begin the shell script with `set -euo pipefail`; require explicit `ROOT_PATH`, `MODEL_PATH`, benchmark, GPU, answers path, and log path. Forward exit status and never overwrite a complete answer file unless `--force` is explicit.

- [ ] **Step 6: Run all CPU tests and commit**

Run: `PYTHONDONTWRITEBYTECODE=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -m unittest discover -s tests -p 'test_evidence_gap_*.py' -v`

```bash
git add cvsearch/evidence_gap/io.py cvsearch/evidence_gap/method.py cvsearch/perform_EGSearch.py cvsearch/run_eval_evidence_gap.sh tests/test_evidence_gap_io.py tests/test_evidence_gap_method.py
git commit -m "feat: add resumable evidence-search runner"
```

### Task 8: Run branch smoke, development calibration, and frozen evaluation

**Files:**
- Create: `reproduction/evidence_gap/configs/minimal_v1.json`
- Create: `reproduction/evidence_gap/configs/frozen_v1.json` for the first development-approved configuration; any later revision uses the next explicit integer filename.
- Create: `reproduction/evidence_gap/scores.json`
- Create: `reproduction/evidence_gap/report.md`

**Interfaces:**
- Consumes the Task 7 CLI and official evaluators.
- Produces exact answer counts, per-benchmark accuracy/cost, paired differences, stop types, fallback rates, and configuration hashes.

- [ ] **Step 1: Re-run baseline gates**

Run existing static tests, score retained gate-0.8 outputs, and reconstruct gate 0.6 in memory. Expected envelope: `87.43 / 76.625 / 76.75`.

- [ ] **Step 2: Run seven branch smoke cases**

Use V* ordinals `8,0,116`, HR-4K `1,31`, and HR-8K `4,31`. For each, run rerank disabled then enabled. Expected: exit 0, evaluator-compatible output, no policy-visible truth keys, no duplicate observation, finite score components, and a nonempty trace.

- [ ] **Step 3: Run the stratified 30-sample-per-benchmark diagnostic**

Select by benchmark category and retained `search_mode` without using correctness labels. Compare runtime, candidate order, root/search selection, and fallback rates. Diagnose any crash or malformed trace before reading accuracy.

- [ ] **Step 4: Freeze the deterministic development split and tune one factor at a time**

Use only the 20% source-grouped dev bucket. Run a coarse grid for `alpha`, `beta`, and root fallback tolerance; keep CLIP and base MLLM revisions fixed. Reject a configuration if any dataset regresses by more than one answer unit or cost exceeds the declared hard budget.

- [ ] **Step 5: Run holdout once for the chosen configuration**

Report paired McNemar on correctness and source-group bootstrap intervals. Accept the configuration only if the aggregate improves and no dataset shows a material regression. Do not revise thresholds from holdout errors.

- [ ] **Step 6: Launch frozen full evaluation on three GPUs**

Run V* on GPU 0, HR-4K on GPU 1, and HR-8K on GPU 2 with the same frozen config. Monitor process liveness, GPU allocation, partial-line growth, log growth, and elapsed time; a long sample is advisory and not auto-killed before the hard timeout.

- [ ] **Step 7: Apply the research gate and choose the next module**

If all scores beat `87.43 / 76.625 / 76.75`, proceed to cost tuning and full `ZOOM/SPLIT/EXPAND/BACKTRACK` ablations. If not, use development traces to identify exactly one failing category:

- ranking failure -> inspect positive-evidence rank and main/augmented/visual ablation;
- root/search regression -> calibrate answer stability/support on dev;
- relationship failure -> implement focus/context `EXPAND` next;
- exhausted search -> enable explicit `SPLIT/ZOOM` budgets;
- false stop -> replace CLIP support with a cross-checkpoint verifier before changing thresholds.

- [ ] **Step 8: Write and validate the report, then commit tracked artifacts**

The report includes config/checkpoint hashes, sample counts, official scores, baseline envelope, per-category deltas, MLLM calls, pixels, p50/p95 latency, certified/forced rates, fallbacks, and known approximations. Run `git diff --check` and verify no raw image, model, answer JSONL, or log is staged.

```bash
git add reproduction/evidence_gap/configs reproduction/evidence_gap/scores.json reproduction/evidence_gap/report.md
git commit -m "experiments: evaluate evidence-gap search"
```

## Plan Self-Review

- Spec coverage: input leakage, dual baseline, no-pruning rank, answer stability, history/termination, budget, trace, resume, staged actions, and frozen evaluation each map to a task.
- Type consistency: all later tasks consume the exact symbols declared in Tasks 1–7.
- Scope: the first executable milestone intentionally defers expensive action adapters and cross-checkpoint verification until Task 8 identifies they are needed; the interfaces and decision gate are explicit rather than implicit unfinished work.
- Placeholder scan: no unspecified implementation steps or unresolved names remain.
