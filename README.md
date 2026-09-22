# Q-AVS

Query-aware adaptive visual search for high-resolution images.

Q-AVS saves a whole-image answer, ranks candidate regions using the question,
adapts the observation to missing evidence, and accepts an answer only after
independent visual verification. The generator and verifier use different frozen
checkpoints. When search ends without an accepted answer, it returns the saved
whole-image answer.

This repository contains the core search implementation and a single-image
inference command. The [method appendix](docs/appendix.md) documents the prompts,
evidence rules and fixed configuration. Its source is the local method draft
dated 2026-09-07; alignment with the latest
[Overleaf manuscript](https://www.overleaf.com/project/6a85563f9b46f95b25cba69d)
has not yet been verified because the project requires access.

## Installation

Use Python 3.11 or newer and a CUDA installation compatible with PyTorch 2.7.
Install the code and its inference dependencies in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SAM 3 and LLaVA are installed from pinned upstream revisions. Model checkpoints
and the spaCy English model are supplied separately, using paths outside this
repository. The model adapters support InternVL2.5, LLaVA-OneVision and Qwen VL.
The local method draft specifies InternVL2.5-8B as generator and
LLaVA-OneVision-7B as verifier for its efficiency configuration.

## Inference

Provide one image, its question and the visible answer choices:

```bash
python -m qavs \
  --image /path/to/image.jpg \
  --question "What color is the man's hat?" \
  --options red blue green black \
  --generator /path/to/InternVL2_5-8B \
  --verifier /path/to/llava-onevision-qwen2-7b-ov \
  --sam /path/to/sam3.pt \
  --clip /path/to/clip-vit-large-patch14 \
  --spacy /path/to/en_core_web_sm \
  --device cuda:0
```

The command prints a JSON answer and its zero-based option index. For binary
existence questions, use `--options Yes No`; the output is `yes` or `no`.
`--verifier-device` places the verifier on a separate device when needed.

[defaults.json](qavs/defaults.json) supplies the fixed search configuration.
The full-image shortcut requires a valid `Support` verdict, absolute support
and a margin over the next option. Local acceptance additionally requires
independent target grounding and distinct confirming views. Accepting `No`
requires the candidate, spatial coverage and refutation checks described in
the appendix.

## Code

| Location | Purpose |
| --- | --- |
| `qavs/__main__.py` | Single-image inference and checkpoint identity checks |
| `qavs/independent_search/` | Proposals, evidence grounding, bundles and answer acceptance |
| `qavs/evidence_gap/` | Question planning, region ranking, observation actions and search state |
| `qavs/models/` | Model adapters and semantic region construction |
| `docs/appendix.md` | Method details and fixed prompts |
| `tests/` | Functional checks using synthetic inputs and model doubles |

Run the functional checks without downloading model checkpoints:

```bash
pip install -e '.[test]'
python -m pytest -q
```

## Acknowledgments

The proposal frontend and model adapters build on
[CVSearch](https://github.com/liliupeng28/ICML26-CVSearch),
[ZoomEye](https://github.com/om-ai-lab/ZoomEye),
[SAM 3](https://github.com/facebookresearch/sam3) and
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).
