# CVSearch Main-Results Reproduction Design

## Objective

Reproduce the six headline accuracy results for the Qwen2.5-VL-7B backbone in Table 1 of *CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception*: direct-answer and CVSearch results on V* Bench, HR-Bench 4K, and HR-Bench 8K.

## Scope

The reproduction covers one backbone and three high-resolution benchmarks:

| Benchmark | Paper baseline | Paper CVSearch |
| --- | ---: | ---: |
| V* Bench | 71.2 | 90.1 |
| HR-Bench 4K | 68.8 | 76.6 |
| HR-Bench 8K | 65.3 | 75.6 |

FineRS-4K, MME-RealWorld-Lite, TreeBench, alternative backbones, ablations, and throughput comparisons are outside this first reproduction milestone.

## Execution Architecture

The existing official evaluation implementation remains the source of truth. A dedicated Python environment is created outside the Git worktree. The cached Qwen2.5-VL-7B checkpoint is reused, while the three benchmark archives, SAM 3 checkpoint, and spaCy model are downloaded from the repositories linked by the project README and verified against their published SHA-256 identifiers.

Each H200 GPU owns one benchmark. On each GPU, a direct-answer run is followed by the corresponding CVSearch run so model and benchmark placement stay isolated:

- GPU 0: V* Bench baseline, then V* Bench CVSearch.
- GPU 1: HR-Bench 4K baseline, then HR-Bench 4K CVSearch.
- GPU 2: HR-Bench 8K baseline, then HR-Bench 8K CVSearch.

The official evaluation scripts convert the six JSONL answer files into accuracy metrics. A final comparison report records paper value, reproduced value, absolute percentage-point difference, and the observed CVSearch gain.

## Data Flow

1. Download and checksum benchmark/model archives.
2. Extract artifacts into project-local `datasets/` and `models/` directories.
3. Validate expected annotation files, images, checkpoint, and spaCy metadata.
4. Import all runtime modules and load the Qwen, SAM 3, and spaCy components.
5. Run one-sample direct-answer and CVSearch smoke tests without altering the official full annotations.
6. Run all six full evaluations, preserving raw JSONL and logs.
7. Score each output with the official evaluator and generate the comparison report.

## Error Handling

Every download must match the LFS pointer SHA-256 before extraction. Every experiment must preserve its log and exit code. A failed run is diagnosed from the complete traceback and is not silently retried. Compatibility changes are allowed only after reproducing and locating the root cause; any such change must be minimal, tested on the smoke sample, and documented in the final report.

Incomplete JSONL files are not scored as complete results. The expected answer count is taken from each official annotation file and compared with the produced JSONL line count before evaluation.

## Validation and Success Criteria

The reproduction is successful when all of the following hold:

- All six experiment processes exit with code 0.
- Each answer file contains exactly the benchmark's expected number of samples.
- Each reproduced accuracy is within 5.0 percentage points of its paper value.
- CVSearch improves over the direct-answer baseline on all three benchmarks.
- The reproduced CVSearch gain on each benchmark is directionally consistent with Table 1.
- Environment versions, commands, raw answers, logs, scores, and comparison data are retained under `reproduction/`.

The target is deterministic inference (`do_sample=False` in the official wrapper). The 5.0-point tolerance accounts for checkpoint revisions, library/kernel differences, and minor nondeterminism while enforcing the user's requirement that results not deviate materially.

## Deliverables

- Six raw answer JSONL files.
- Six inference logs and three evaluator outputs per method.
- Machine-readable score comparison data.
- A Markdown reproduction report containing environment details, artifact checksums, commands, results, deviations, anomalies, and the final reproducibility verdict.
