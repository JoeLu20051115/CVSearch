import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cvsearch.evidence_gap.pdf_types import PDFSearchConfig
from cvsearch.eval.pdf_trace_audit import audit_pdf_trace, audit_pdf_traces
from cvsearch.evidence_gap.provenance import canonical_sha256
from cvsearch.perform_PDFSearch import run_pdf_sample
from tests.test_pdf_types import full_config
from tests.test_perform_pdf_search import (
    FakeClip,
    FakeGenerator,
    FakeVerifier,
    fake_cvsearch,
)


def make_trace():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        Image.new("RGB", (8, 8), "white").save(folder / "image.png")
        _, trace = run_pdf_sample(
            original_annotation={
                "question": "Which sign is visible?", "options": ["left", "right"],
                "answer_type": "logits_match", "input_image": "image.png",
            },
            image_folder=folder, ic_examples={},
            config=PDFSearchConfig.from_mapping(full_config()),
            sam_model=object(), generator_model=FakeGenerator(),
            verifier_model=FakeVerifier(), nlp_model=object(),
            clip_scorer=FakeClip(), cvsearch_fn=fake_cvsearch,
            generator_checkpoint_sha256="a" * 64,
            verifier_checkpoint_sha256="b" * 64,
        )
    return trace


class PDFTraceAuditTest(unittest.TestCase):
    def test_valid_full_trace_proves_planner_top3_uncertainty_and_verifier_activity(self):
        report = audit_pdf_trace(make_trace(), require_operational=True)
        self.assertEqual(report["ranking_candidates"], 2)
        self.assertEqual(report["state_evaluations"], 2)
        self.assertEqual(report["actions"]["SPLIT"], 1)
        self.assertEqual(report["verifier_fallback_states"], 0)
        self.assertEqual(report["termination"], "CERTIFIED_STOP")

    def test_rejects_fake_top3_hash_mismatch_and_evaluator_metadata(self):
        for mutate in (
            lambda trace: trace["joint_ranking"][0].update({"top_k_augmented": 2}),
            lambda trace: trace["candidate_factory"].update({"collector_sha256": "0" * 64}),
            lambda trace: trace.update({"ground_truth": "secret"}),
        ):
            with self.subTest(mutate=mutate):
                trace = copy.deepcopy(make_trace())
                mutate(trace)
                with self.assertRaises(ValueError):
                    audit_pdf_trace(trace, require_operational=True)

    def test_operational_audit_rejects_p0_only_candidate_factory(self):
        trace = copy.deepcopy(make_trace())
        collector = trace["candidate_factory"]["collector"]
        for snapshot in collector["snapshots"]:
            snapshot["event"] = "p0_selected"
        trace["candidate_factory"]["collector_sha256"] = canonical_sha256(collector)

        with self.assertRaisesRegex(ValueError, "root-to-leaf tree"):
            audit_pdf_trace(trace, require_operational=True)

    def test_aggregate_keeps_action_stop_and_fallback_counts(self):
        trace = make_trace()
        report = audit_pdf_traces([trace, copy.deepcopy(trace)], require_operational=True)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["actions"]["SPLIT"], 2)
        self.assertEqual(report["terminations"]["CERTIFIED_STOP"], 2)


if __name__ == "__main__":
    unittest.main()
