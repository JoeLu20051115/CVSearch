import json
import math
import unittest

from qavs.evidence_gap.pdf_types import (
    ActionName,
    CandidateDescriptor,
    EvidenceGapScores,
    ModelFingerprint,
    PDFSearchConfig,
    SearchStateRecord,
    TerminationType,
    validate_independent_checkpoints,
)


def full_config():
    return {
        "schema_version": 1,
        "method": "pdf_faithful_v1",
        "profile": "full",
        "ranking": {
            "alpha": 0.6,
            "beta": 0.5,
            "visual_lambda": 0.5,
            "top_k_augmented": 3,
            "main_query": True,
            "augmented_query": True,
            "complexity": True,
            "edge_density": True,
        },
        "modules": {
            "structured_planner": True,
            "zoom": True,
            "split": True,
            "expand": True,
            "next": True,
            "reanswer": True,
            "independent_verifier": True,
            "backtracking": True,
            "certified_stop": True,
        },
        "budget": {
            "max_steps": 8,
            "max_model_calls": 96,
            "max_processed_pixels": 600000000,
        },
        "controller": {
            "stall_patience": 2,
            "min_progress": 0.02,
            "max_gap_for_stop": 0.2,
            "min_answer_confidence": 0.65,
            "min_support_avg": 0.7,
            "min_support_min": 0.5,
        },
    }


class PDFSearchConfigTest(unittest.TestCase):
    def test_full_profile_requires_every_module_and_exact_top_three(self):
        config = PDFSearchConfig.from_mapping(full_config())
        self.assertEqual(config.ranking.top_k_augmented, 3)
        self.assertTrue(all(config.modules.to_dict().values()))
        self.assertEqual(config.to_dict(), full_config())
        self.assertEqual(json.loads(config.to_json()), full_config())

        for key in full_config()["modules"]:
            invalid = full_config()
            invalid["modules"][key] = False
            with self.subTest(module=key), self.assertRaisesRegex(ValueError, key):
                PDFSearchConfig.from_mapping(invalid)

        invalid = full_config()
        invalid["ranking"]["top_k_augmented"] = 2
        with self.assertRaisesRegex(ValueError, "top_k_augmented"):
            PDFSearchConfig.from_mapping(invalid)

    def test_core_profile_can_disable_one_declared_component(self):
        config = full_config()
        config["profile"] = "core"
        config["modules"]["split"] = False
        parsed = PDFSearchConfig.from_mapping(config)
        self.assertFalse(parsed.modules.split)

    def test_rejects_unknown_missing_boolean_and_nonfinite_values(self):
        invalid = full_config()
        invalid["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "exact keys"):
            PDFSearchConfig.from_mapping(invalid)

        invalid = full_config()
        del invalid["controller"]["min_support_min"]
        with self.assertRaisesRegex(ValueError, "exact keys"):
            PDFSearchConfig.from_mapping(invalid)

        invalid = full_config()
        invalid["modules"]["zoom"] = 1
        with self.assertRaisesRegex(TypeError, "zoom"):
            PDFSearchConfig.from_mapping(invalid)

        for value in (math.nan, math.inf, -0.1, 1.1):
            invalid = full_config()
            invalid["ranking"]["alpha"] = value
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                PDFSearchConfig.from_mapping(invalid)


class PDFRecordTest(unittest.TestCase):
    def test_gap_scores_are_independent_finite_probabilities(self):
        scores = EvidenceGapScores(zoom=0.1, split=0.2, expand=0.3, next=0.4)
        self.assertEqual(
            scores.to_dict(), {"zoom": 0.1, "split": 0.2, "expand": 0.3, "next": 0.4}
        )
        with self.assertRaises(ValueError):
            EvidenceGapScores(zoom=0.1, split=0.2, expand=0.3, next=math.nan)

    def test_candidate_and_state_are_immutable_and_strict_json(self):
        candidate = CandidateDescriptor(
            canonical_key="box:1", sibling_group="root", native_ordinal=0,
            bbox_original=(1.0, 2.0, 30.0, 40.0), depth=1, render_level=0,
        )
        state = SearchStateRecord(
            state_id=0,
            focus_keys=(candidate.canonical_key,),
            path_keys=(candidate.canonical_key,),
            context_keys=(),
            visited_keys=(candidate.canonical_key,),
            observation_keys=("box:1@root",),
            remaining_steps=8,
            remaining_model_calls=96,
            remaining_pixels=600000000,
        )
        with self.assertRaises(Exception):
            candidate.depth = 2
        self.assertEqual(json.loads(json.dumps(candidate.to_dict())), candidate.to_dict())
        self.assertEqual(json.loads(json.dumps(state.to_dict())), state.to_dict())

    def test_action_and_termination_names_are_not_conflated(self):
        self.assertEqual(
            [item.value for item in ActionName],
            ["ZOOM", "SPLIT", "EXPAND", "NEXT", "BACKTRACK"],
        )
        self.assertEqual(
            [item.value for item in TerminationType],
            ["ACCEPTED_STOP", "CERTIFIED_STOP", "FORCED_RETURN"],
        )

    def test_verifier_checkpoint_must_be_independent(self):
        generator = ModelFingerprint("llava", "a" * 64)
        verifier = ModelFingerprint("qwen", "b" * 64)
        validate_independent_checkpoints(generator, verifier)
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_independent_checkpoints(generator, ModelFingerprint("alias", "a" * 64))


if __name__ == "__main__":
    unittest.main()
