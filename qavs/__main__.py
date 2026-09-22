"""Run Q-AVS on one image and a question with visible answer options."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys


def checkpoint_identity(path: Path) -> str:
    """Identify checkpoint contents, including aliases and copied weights."""
    path = path.expanduser().resolve(strict=True)
    config = path / "config.json"
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    if not config.is_file() or not weights:
        raise ValueError(f"checkpoint needs config.json and model weights: {path}")
    digest = hashlib.sha256(config.read_bytes())
    for weight in weights:
        with weight.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--sam", type=Path, required=True, help="SAM 3 checkpoint file")
    parser.add_argument("--clip", type=Path, required=True, help="local CLIP checkpoint directory")
    parser.add_argument("--spacy", required=True, help="installed spaCy model name or local directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verifier-device", default=None)
    args = parser.parse_args(argv)
    if not args.question.strip() or len(args.options) < 2:
        parser.error("provide a nonempty question and at least two answer options")
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")

    from qavs.independent_search.config import IndependentSearchConfig
    from qavs.independent_search.semantics import build_option_catalog

    options = [option.strip() for option in args.options]
    binary = len(options) == 2 and {value.casefold() for value in options} == {"yes", "no"}
    policy = {
        "input_image": str(args.image.resolve()), "question": args.question.strip(),
        "options": options, "answer_type": "yes_no" if binary else "logits_match",
    }
    build_option_catalog(policy)
    config = IndependentSearchConfig.from_mapping(
        json.loads(Path(__file__).with_name("defaults.json").read_text())
    )
    generator_id = checkpoint_identity(args.generator)
    verifier_id = checkpoint_identity(args.verifier)
    if generator_id == verifier_id:
        parser.error("generator and verifier must use different checkpoints")

    # Model libraries may print while loading; reserve stdout for the answer.
    with redirect_stdout(sys.stderr):
        import spacy
        import torch
        from qavs.evidence_gap.clip_scorer import ClipScorer
        from qavs.independent_search.method import run_independent_sample
        from qavs.models.modeling_dispatch import detect_model_family, load_search_model
        from qavs.models.modeling_sam3 import sam3_inference

        generator = load_search_model(args.generator, device=args.device)
        verifier = load_search_model(args.verifier, device=args.verifier_device or args.device)
        for wrapper in (generator, verifier):
            wrapper.model.eval().requires_grad_(False)
        sam = sam3_inference(model_path=str(args.sam), device=args.device)
        clip = ClipScorer(model_path=str(args.clip), device=args.device)
        nlp = spacy.load(args.spacy)
        with torch.inference_mode():
            output, _ = run_independent_sample(
                original_annotation=policy, image_folder=args.image.resolve().parent,
                ic_examples={
                    "question_template": (
                        "Question: {}\nIdentify only the visible objects needed to answer "
                        "this question. Do not answer it or guess missing attributes. "
                        "Return exactly: So I need the information about the following "
                        "objects: <comma-separated object descriptions>."
                    ),
                    "question_list": [], "response_list": [],
                },
                config=config, sam_model=sam,
                generator_model=generator, verifier_model=verifier,
                nlp_model=nlp, clip_scorer=clip,
                generator_checkpoint_sha256=generator_id,
                verifier_checkpoint_sha256=verifier_id,
                generator_family=detect_model_family(args.generator),
            )
    result = {"answer": output if binary else options[output]}
    if not binary:
        result["option_index"] = output
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
