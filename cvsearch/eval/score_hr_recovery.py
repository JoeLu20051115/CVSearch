#!/usr/bin/env python3
"""Score one explicitly selected locked HR-Bench recovery split."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cvsearch.evidence_gap.hr_recovery import load_jsonl, score_recovery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("recovery-a", "vault-b"), required=True)
    for resolution in ("4k", "8k"):
        parser.add_argument(f"--method-{resolution}", required=True)
        parser.add_argument(f"--direct-{resolution}", required=True)
        parser.add_argument(f"--cvsearch-{resolution}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = score_recovery(
        args.split,
        method_4k=load_jsonl(args.method_4k),
        method_8k=load_jsonl(args.method_8k),
        direct_4k=load_jsonl(args.direct_4k),
        cvsearch_4k=load_jsonl(args.cvsearch_4k),
        direct_8k=load_jsonl(args.direct_8k),
        cvsearch_8k=load_jsonl(args.cvsearch_8k),
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
