# CVSearch Main-Results Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the Qwen2.5-VL-7B direct-answer and CVSearch accuracies on V* Bench, HR-Bench 4K, and HR-Bench 8K with three H200 GPUs and retain auditable outputs.

**Architecture:** Reuse the cached Qwen checkpoint and clone the compatible `memgen` Conda environment into an external isolated prefix. Download only the three selected benchmark archives plus SAM 3 and spaCy, run one-sample smoke checks, then assign one complete benchmark pair to each GPU and score outputs with the official evaluators.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu118, Transformers 4.57.0, FlashAttention 2.8.3, ModelScope CLI, Qwen2.5-VL-7B, SAM 3, spaCy 3.8.7, NVIDIA H200 NVL.

## Global Constraints

- Preserve the official CVSearch algorithm and default hyperparameters.
- Run deterministic inference with the repository's existing `do_sample=False` behavior.
- Use exactly GPUs 0, 1, and 2; one benchmark per GPU.
- Require six clean process exits and exact answer-file sample counts.
- Require every reproduced accuracy to be within 5.0 percentage points of Table 1 and CVSearch to outperform direct answer on every benchmark.
- Never silently retry a failed experiment; retain its full log and diagnose the root cause first.
- Keep the Python environment outside the Git worktree at `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch`.

---

### Task 1: Isolate generated artifacts from Git

**Files:**
- Create: `.gitignore`
- Create: `reproduction/README.md`

**Interfaces:**
- Consumes: approved design at `docs/superpowers/specs/2026-08-01-cvsearch-main-results-reproduction-design.md`.
- Produces: stable artifact locations used by every later command.

- [ ] **Step 1: Add generated-path exclusions**

Create `.gitignore` with:

```gitignore
artifacts/
datasets/
models/
reproduction/answers/
reproduction/logs/
```

- [ ] **Step 2: Document retained outputs**

Create `reproduction/README.md` describing the six answer paths, six inference logs, evaluator logs, `scores.json`, and `report.md`.

- [ ] **Step 3: Verify ignore behavior**

Run:

```bash
git check-ignore -v --no-index artifacts/probe datasets/probe models/probe reproduction/answers/probe reproduction/logs/probe
git diff --check
```

Expected: all five generated paths are reported and `git diff --check` exits 0.

- [ ] **Step 4: Commit repository metadata**

```bash
git add .gitignore reproduction/README.md docs/superpowers/plans/2026-08-01-cvsearch-main-results-reproduction.md
git commit -m "docs: plan CVSearch main-results reproduction"
```

### Task 2: Build and validate the isolated runtime

**Files:**
- Create outside worktree: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/`
- Test: import smoke command below

**Interfaces:**
- Consumes: `/home/xingrui/storage/miniforge3/envs/memgen` with PyTorch 2.7.1, FlashAttention 2.8.3, spaCy 3.8.7, and CUDA access.
- Produces: `/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python` with all official inference imports available.

- [ ] **Step 1: Clone the compatible base environment**

```bash
/mnt/data3/data_xingrui/miniforge3/bin/conda create -y -p /mnt/data3/data_xingrui/lueq/.venvs/cvsearch --clone /home/xingrui/storage/miniforge3/envs/memgen
```

Expected: Conda exits 0 and the target Python reports version 3.11.

- [ ] **Step 2: Install only missing inference dependencies**

```bash
uv pip install --python /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python transformers==4.57.0 matplotlib==3.10.9 scikit-learn==1.8.0 scikit-image==0.25.2 hydra-core==1.3.2 iopath==0.1.10 timm==1.0.27 ftfy==6.3.1 tensordict==0.1.2 torchmetrics==1.9.0 submitit==1.5.4
```

Expected: installation exits 0 without replacing PyTorch 2.7.1 or FlashAttention 2.8.3.

- [ ] **Step 3: Verify versions and CUDA**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import torch, transformers, flash_attn, spacy; from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration; print(torch.__version__, transformers.__version__, flash_attn.__version__, spacy.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: versions include `2.7.1`, `4.57.0`, `2.8.3`, `3.8.7`, followed by `True` and `NVIDIA H200 NVL`.

- [ ] **Step 4: Import the official entry point**

Run from `cvsearch/`:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import perform_CVSearch; print('CVSearch imports OK')"
```

Expected: `CVSearch imports OK` and exit 0.

### Task 3: Download, checksum, and extract artifacts

**Files:**
- Create: `artifacts/downloads/hr_data/{vstar,hr-bench_4k,hr-bench_8k}.tar.gz`
- Create: `artifacts/downloads/sam3/{sam3,en_core_web_sm-3.8.0}.tar.gz`
- Create: `datasets/hr_data/{vstar,hr-bench_4k,hr-bench_8k}/`
- Create: `models/sam3/`
- Create: `models/en_core_web_sm-3.8.0/`

**Interfaces:**
- Consumes: ModelScope repositories `llp1995/hr_data` and `llp1995/sam3`.
- Produces: official annotation/image trees plus SAM checkpoint and spaCy model paths accepted by `perform_CVSearch.py`.

- [ ] **Step 1: Download the 12 MB spaCy archive as a transport proof**

```bash
uvx --from 'modelscope>=1.29' modelscope download llp1995/sam3 en_core_web_sm-3.8.0.tar.gz --local-dir artifacts/downloads/sam3
sha256sum artifacts/downloads/sam3/en_core_web_sm-3.8.0.tar.gz
```

Expected SHA-256: `4651b985ecdc408201f5217bc0bc38d4a8a40e8418ada6c58d5a8473ccd76649`.

- [ ] **Step 2: Download selected benchmark archives**

```bash
uvx --from 'modelscope>=1.29' modelscope download --repo-type dataset llp1995/hr_data vstar.tar.gz hr-bench_4k.tar.gz hr-bench_8k.tar.gz --local-dir artifacts/downloads/hr_data
```

Expected: three non-empty archives and exit 0.

- [ ] **Step 3: Download SAM 3**

```bash
uvx --from 'modelscope>=1.29' modelscope download llp1995/sam3 sam3.tar.gz --local-dir artifacts/downloads/sam3
```

Expected: `sam3.tar.gz` is 6,390,731,102 bytes and the command exits 0.

- [ ] **Step 4: Verify all archive hashes**

```bash
sha256sum artifacts/downloads/hr_data/vstar.tar.gz artifacts/downloads/hr_data/hr-bench_4k.tar.gz artifacts/downloads/hr_data/hr-bench_8k.tar.gz artifacts/downloads/sam3/sam3.tar.gz artifacts/downloads/sam3/en_core_web_sm-3.8.0.tar.gz
```

Expected hashes in command order:

```text
469b1d07016cb5cc7e3ceb433b3460c6910804b4a013ca8a4ae0ab5f1030e8b7
f5d6b712681e3c9d0b4a4ca5dc0c5905f9cbca574523fb0e6847bada8f82553c
0bf1ae2cd36d63322d4c9169efef7dcf55afae6bbebb9cea1132a9e66dce0830
6152c6a75fa9dae9b2abbd645c936e2df1efb80c9c6beb21e3fd6057a706caa8
4651b985ecdc408201f5217bc0bc38d4a8a40e8418ada6c58d5a8473ccd76649
```

- [ ] **Step 5: Inspect and extract archives**

```bash
tar -tzf artifacts/downloads/hr_data/vstar.tar.gz | sed -n '1,20p'
tar -tzf artifacts/downloads/sam3/sam3.tar.gz | sed -n '1,20p'
mkdir -p datasets/hr_data models
tar -xzf artifacts/downloads/hr_data/vstar.tar.gz -C datasets/hr_data
tar -xzf artifacts/downloads/hr_data/hr-bench_4k.tar.gz -C datasets/hr_data
tar -xzf artifacts/downloads/hr_data/hr-bench_8k.tar.gz -C datasets/hr_data
tar -xzf artifacts/downloads/sam3/sam3.tar.gz -C models
tar -xzf artifacts/downloads/sam3/en_core_web_sm-3.8.0.tar.gz -C models
```

Expected: extraction exits 0 and creates no files outside the declared directories.

- [ ] **Step 6: Validate artifact structure and sample counts**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import json, pathlib; root=pathlib.Path('datasets/hr_data'); names=['vstar','hr-bench_4k','hr-bench_8k']; print({n: len(json.loads((root/n/f'annotation_{n}.json').read_text())) for n in names})"
find models -type f -name 'sam3.pt' -o -name 'meta.json'
```

Expected: all three counts are positive, exactly one `sam3.pt` is present, and the spaCy model contains `meta.json`.

### Task 4: Run one-sample end-to-end smoke checks

**Files:**
- Create: `reproduction/answers/smoke/direct_answer.jsonl`
- Create: `reproduction/answers/smoke/cvsearch.jsonl`
- Create: `reproduction/logs/smoke-direct.log`
- Create: `reproduction/logs/smoke-cvsearch.log`

**Interfaces:**
- Consumes: validated runtime and artifacts from Tasks 2-3.
- Produces: one valid direct answer and one valid CVSearch answer using V* sample 0.

- [ ] **Step 1: Resolve artifact paths and V* sample count**

```bash
find models -type f -name sam3.pt
find models -type f -name meta.json
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import json; print(len(json.load(open('datasets/hr_data/vstar/annotation_vstar.json'))))"
```

Expected: one checkpoint path, one spaCy model root, and a positive integer `N`.

- [ ] **Step 2: Run direct-answer sample 0**

Run from `cvsearch/`:

```bash
cv_vstar_count=$(/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import json; print(len(json.load(open('/mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data/vstar/annotation_vstar.json'))))")
mkdir -p /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/smoke /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs
CUDA_VISIBLE_DEVICES=0 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark vstar --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/smoke/direct_answer.jsonl --num-chunks "$cv_vstar_count" --chunk-idx 0 --direct-answer 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/smoke-direct.log
```

Expected: exit 0 and exactly one JSONL record with an `output` field.

- [ ] **Step 3: Run CVSearch sample 0**

Run from `cvsearch/`:

```bash
cv_vstar_count=$(/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import json; print(len(json.load(open('/mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data/vstar/annotation_vstar.json'))))")
CUDA_VISIBLE_DEVICES=0 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark vstar --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/smoke/cvsearch.jsonl --num-chunks "$cv_vstar_count" --chunk-idx 0 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/smoke-cvsearch.log
```

Expected: exit 0 and exactly one JSONL record containing `output`, `search_mode`, and `root_ans_conf`.

- [ ] **Step 4: Validate smoke outputs**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import json; files=['reproduction/answers/smoke/direct_answer.jsonl','reproduction/answers/smoke/cvsearch.jsonl']; rows=[json.loads(open(f).readline()) for f in files]; assert all('output' in r for r in rows); assert 'search_mode' in rows[1] and 'root_ans_conf' in rows[1]; print('smoke outputs OK')"
```

Expected: `smoke outputs OK`.

### Task 5: Run the six full evaluations on three GPUs

**Files:**
- Create: `reproduction/answers/qwen2.5-vl-7b/{vstar,hr-bench_4k,hr-bench_8k}/{direct_answer,cvsearch}.jsonl`
- Create: `reproduction/logs/{vstar,hr-bench_4k,hr-bench_8k}-{direct,cvsearch}.log`

**Interfaces:**
- Consumes: the exact smoke-tested command with `--num-chunks 1 --chunk-idx 0`.
- Produces: six complete answer files and six process exit codes.

- [ ] **Step 1: Start GPU 0 V* baseline then CVSearch**

Run from `cvsearch/`:

```bash
mkdir -p /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/vstar /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark vstar --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/vstar/direct_answer.jsonl --num-chunks 1 --chunk-idx 0 --direct-answer 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/vstar-direct.log
CUDA_VISIBLE_DEVICES=0 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark vstar --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/vstar/cvsearch.jsonl --num-chunks 1 --chunk-idx 0 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/vstar-cvsearch.log
```

Expected: both commands exit 0.

- [ ] **Step 2: Start GPU 1 HR-Bench 4K baseline then CVSearch**

Run from `cvsearch/`:

```bash
mkdir -p /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/hr-bench_4k /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark hr-bench_4k --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/hr-bench_4k/direct_answer.jsonl --num-chunks 1 --chunk-idx 0 --direct-answer 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/hr-bench_4k-direct.log
CUDA_VISIBLE_DEVICES=1 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark hr-bench_4k --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/hr-bench_4k/cvsearch.jsonl --num-chunks 1 --chunk-idx 0 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/hr-bench_4k-cvsearch.log
```

Expected: both commands exit 0.

- [ ] **Step 3: Start GPU 2 HR-Bench 8K baseline then CVSearch**

Run from `cvsearch/`:

```bash
mkdir -p /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/hr-bench_8k /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=2 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark hr-bench_8k --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/hr-bench_8k/direct_answer.jsonl --num-chunks 1 --chunk-idx 0 --direct-answer 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/hr-bench_8k-direct.log
CUDA_VISIBLE_DEVICES=2 /mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python perform_CVSearch.py --root-path / --model-path /home/xingrui/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --benchmark hr-bench_8k --annotation_path /mnt/data3/data_xingrui/lueq/CVSearch/datasets/hr_data --sam-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/sam3/sam3.pt --nlp-model-path /mnt/data3/data_xingrui/lueq/CVSearch/models/en_core_web_sm-3.8.0 --answers-file /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/answers/qwen2.5-vl-7b/hr-bench_8k/cvsearch.jsonl --num-chunks 1 --chunk-idx 0 2>&1 | tee /mnt/data3/data_xingrui/lueq/CVSearch/reproduction/logs/hr-bench_8k-cvsearch.log
```

Expected: both commands exit 0.

- [ ] **Step 4: Monitor all three workers**

Every 30-60 seconds, verify process liveness, GPU allocation, answer-file line growth, elapsed time, and log growth. Treat 90 seconds without answer growth as advisory because a single CVSearch sample can legitimately take longer; do not kill unless the agreed six-hour hard timeout is exceeded.

Expected: all workers reach clean exit without OOM, traceback, or truncated JSON.

- [ ] **Step 5: Verify exact output counts**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python -c "import json, pathlib; root=pathlib.Path('datasets/hr_data'); ans=pathlib.Path('reproduction/answers/qwen2.5-vl-7b'); names=['vstar','hr-bench_4k','hr-bench_8k']; expected={n:len(json.loads((root/n/f'annotation_{n}.json').read_text())) for n in names}; actual={(n,m):sum(1 for _ in (ans/n/f'{m}.jsonl').open()) for n in names for m in ['direct_answer','cvsearch']}; print(expected); print(actual); assert all(actual[n,m]==expected[n] for n in names for m in ['direct_answer','cvsearch'])"
```

Expected: assertion passes.

### Task 6: Score, compare, and report

**Files:**
- Create: `reproduction/logs/eval-*.log`
- Create: `reproduction/scores.json`
- Create: `reproduction/report.md`

**Interfaces:**
- Consumes: six complete JSONL files and official evaluators in `cvsearch/eval/`.
- Produces: the final six-number comparison and reproducibility verdict.

- [ ] **Step 1: Score V* outputs**

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python cvsearch/eval/eval_results_vstar.py --answers-file reproduction/answers/qwen2.5-vl-7b/vstar/direct_answer.jsonl
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python cvsearch/eval/eval_results_vstar.py --answers-file reproduction/answers/qwen2.5-vl-7b/vstar/cvsearch.jsonl
```

Expected: each evaluator reports the full sample count and overall accuracy.

- [ ] **Step 2: Score HR-Bench outputs**

Run:

```bash
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python cvsearch/eval/eval_results_hr-bench.py --answers-file reproduction/answers/qwen2.5-vl-7b/hr-bench_4k/direct_answer.jsonl
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python cvsearch/eval/eval_results_hr-bench.py --answers-file reproduction/answers/qwen2.5-vl-7b/hr-bench_4k/cvsearch.jsonl
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python cvsearch/eval/eval_results_hr-bench.py --answers-file reproduction/answers/qwen2.5-vl-7b/hr-bench_8k/direct_answer.jsonl
/mnt/data3/data_xingrui/lueq/.venvs/cvsearch/bin/python cvsearch/eval/eval_results_hr-bench.py --answers-file reproduction/answers/qwen2.5-vl-7b/hr-bench_8k/cvsearch.jsonl
```

Expected: each evaluator reports FSP, FCP, and total accuracy.

- [ ] **Step 3: Create `scores.json`**

Record for each benchmark and method: paper accuracy, reproduced accuracy, signed difference, absolute difference, expected sample count, actual sample count, and command exit status.

- [ ] **Step 4: Apply the acceptance gate**

Assert in one Python command that every absolute difference is at most 5.0 and that every reproduced CVSearch score exceeds its reproduced direct-answer score.

Expected: the command prints `REPRODUCTION VERIFIED` and exits 0.

- [ ] **Step 5: Write the ARS-compatible report**

Create `reproduction/report.md` with a Material Passport, environment and hardware versions, source commit, artifact checksums, exact commands, result table, deviations, anomaly log, fallacy/interpretation notes, and `Verification Status: VERIFIED` only if Step 4 passes.

- [ ] **Step 6: Verify deliverables and repository state**

```bash
git diff --check
git status --short --branch
find reproduction -maxdepth 4 -type f -printf '%p %s bytes\n' | sort
```

Expected: report and score files are present, ignored raw artifacts remain on disk, and no unrelated files are staged.

- [ ] **Step 7: Commit the compact reproduction record**

```bash
git add reproduction/README.md reproduction/scores.json reproduction/report.md
git commit -m "docs: record CVSearch main-results reproduction"
```
