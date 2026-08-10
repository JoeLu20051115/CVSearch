import hashlib
import tempfile
import unittest

from PIL import Image

from cvsearch.eval.phase7_uncertainty_confirmation import (
    CONFIRMATION_BACKGROUND_RGB,
    CONFIRMATION_SEPARATOR_PIXELS,
    VSTAR_CONFIRMATION_TEMPLATES,
    compose_confirmation_view,
    confirmation_prompt_material,
    render_expand_action_views,
    render_zoom_action_views,
)
import tests.test_evidence_gap_zoom_observation as zoom_test
from tests.test_phase4_expand_oracle import PairFactory


class Phase7ConfirmationRendererTest(unittest.TestCase):
    def test_exact_candidate_top_broader_bottom_and_fixed_separator(self):
        candidate = Image.new("RGB", (3, 2), (255, 0, 0))
        broader = Image.new("RGB", (2, 4), (0, 0, 255))

        rendered, audit = compose_confirmation_view(candidate, broader=broader)

        self.assertEqual(rendered.size, (3, 14))
        self.assertEqual(rendered.crop((0, 0, 3, 2)).tobytes(), candidate.tobytes())
        self.assertEqual(
            rendered.crop((0, 10, 2, 14)).tobytes(), broader.tobytes(),
        )
        separator = rendered.crop((0, 2, 3, 10))
        self.assertEqual(
            set(separator.getdata()), {tuple(CONFIRMATION_BACKGROUND_RGB)},
        )
        self.assertEqual(audit["separator_pixels"], CONFIRMATION_SEPARATOR_PIXELS)
        self.assertEqual(audit["candidate_offset_xy"], [0, 0])
        self.assertEqual(audit["broader_offset_xy"], [0, 10])
        self.assertFalse(audit["whole_image_fallback"])

    def test_equal_views_reject_and_whole_image_fallback_is_letterboxed(self):
        candidate = Image.new("RGB", (6, 4), (255, 0, 0))
        with self.assertRaisesRegex(ValueError, "distinct"):
            compose_confirmation_view(candidate, broader=candidate.copy())

        source = Image.new("RGB", (8, 2), (0, 255, 0))
        rendered, audit = compose_confirmation_view(
            candidate, broader=None, source_image=source,
        )
        self.assertTrue(audit["whole_image_fallback"])
        self.assertEqual(audit["broader_size"], [6, 4])
        bottom = rendered.crop((0, 12, 6, 16))
        self.assertEqual(bottom.getpixel((0, 0)), tuple(CONFIRMATION_BACKGROUND_RGB))
        self.assertEqual(bottom.getpixel((3, 2)), (0, 255, 0))

        with self.assertRaisesRegex(ValueError, "source_image"):
            compose_confirmation_view(candidate, broader=None)

    def test_prompt_material_is_frozen_answer_free_and_dataset_independent(self):
        question = "What color is the object?"
        options = ["red", "blue", "green"]
        vstar = confirmation_prompt_material("logits_match", question, options)
        self.assertEqual(len(vstar["prompts"]), 3)
        self.assertEqual(len(VSTAR_CONFIRMATION_TEMPLATES), 3)
        self.assertEqual(len(set(vstar["prompt_sha256"])), 3)
        for prompt, digest in zip(vstar["prompts"], vstar["prompt_sha256"]):
            self.assertIn(question, prompt)
            self.assertNotIn("candidate answer", prompt.casefold())
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), digest)

        blocks = ["A. red\nB. blue"] * 4
        hr = confirmation_prompt_material("option_list", question, blocks)
        self.assertEqual(len(hr["prompts"]), 4)
        self.assertEqual(len(set(hr["prompt_sha256"])), 1)
        self.assertTrue(all(
            prompt.endswith("Answer the option letter directly.")
            for prompt in hr["prompts"]
        ))

        with self.assertRaisesRegex(ValueError, "answer type"):
            confirmation_prompt_material("unknown", question, options)

    def test_expand_replay_matches_both_frozen_runtime_view_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = PairFactory(directory)
            _, enabled, _ = factory.pair("vstar")
            plan = enabled["method_trace"]["steps"][0]["expand_audit"][
                "batch_result"
            ]["batch_plan"]
            with Image.open(factory.image_path) as opened:
                candidate, broader, audit = render_expand_action_views(
                    opened.convert("RGB"), plan,
                )

        self.assertEqual(
            audit["candidate_view_sha256"],
            plan["candidate_observation"]["view_sha256"],
        )
        self.assertEqual(
            audit["broader_view_sha256"],
            plan["current_observation"]["view_sha256"],
        )
        self.assertNotEqual(candidate.tobytes(), broader.tobytes())

    def test_zoom_replay_matches_native_336_and_112_view_hashes(self):
        case = zoom_test.CoordinateZoomBatchTest()
        result, _, model, source = case._run_batch(answer_type="logits_match")
        plan = result.batch_plan
        candidate, broader, audit = render_zoom_action_views(
            source, plan, model,
        )

        self.assertEqual(model.view_size, plan["base_view_size"])
        self.assertEqual(
            audit["candidate_view_sha256"],
            plan["candidate_observation"]["view_sha256"],
        )
        self.assertEqual(
            audit["broader_view_sha256"],
            plan["current_observation"]["view_sha256"],
        )
        self.assertNotEqual(candidate.size, broader.size)


if __name__ == "__main__":
    unittest.main()
