import unittest

from cvsearch.eval.phase12_generated_query_lazy_search import (
    AdaptivePatch,
    all_lazy_rule_keys,
    choose_backtrack_patch,
    fuse_query_relevance,
    generate_lazy_children,
    parse_localization_queries,
    project_lazy_candidate,
    sanitize_localization_queries,
    select_lazy_candidate,
)


def p0(output=0, confidence=0.2):
    return {
        "action": "P0", "output": output,
        "p0_stability": {"confidence": confidence},
    }


def candidate(output=1, confidence=0.8, path_consensus=1.0):
    return {
        "action": "LAZY", "feasible": True, "output": output,
        "stability": {"confidence": confidence},
        "path_consensus": path_consensus,
        "view_sha256": ["1" * 64, "2" * 64],
        "rank_sha256": "3" * 64,
        "query_sha256": "4" * 64,
    }


class GeneratedQueryTests(unittest.TestCase):
    def test_parses_unique_loc_lines_without_answer_content_fallback(self):
        raw = "LOC: blue road sign\n2. LOC: bus beside sign\nLOC: blue road sign\nLOC: sign text area"
        self.assertEqual(parse_localization_queries(raw), (
            "blue road sign", "bus beside sign", "sign text area",
        ))
        with self.assertRaisesRegex(ValueError, "LOC"):
            parse_localization_queries("The answer is A.")

    def test_fuses_main_with_patch_local_top_three_augmentations(self):
        values = fuse_query_relevance(
            [0.2, 0.8],
            [[0.9, 0.7, 0.5, 0.1], [0.1, 0.2]],
        )
        self.assertAlmostEqual(values[0], 0.45)
        self.assertAlmostEqual(values[1], 0.475)

    def test_sanitizer_removes_attributes_not_present_in_original_question(self):
        queries = sanitize_localization_queries(
            "What is the color of the guard's glove?",
            (
                "guard's black leather glove in foreground",
                "the glove color region",
            ),
        )
        joined = " ".join(queries).casefold()
        self.assertIn("glove", joined)
        self.assertNotIn("black", joined)
        self.assertNotIn("leather", joined)
        self.assertNotIn("foreground", joined)


class LazyPatchTests(unittest.TestCase):
    def test_children_are_unique_contained_original_coordinate_patches(self):
        root = AdaptivePatch((), (10, 20, 110, 100))
        children = generate_lazy_children(root)
        self.assertEqual([child.path for child in children], [
            (0,), (1,), (2,), (3,),
        ])
        self.assertEqual(len({child.box for child in children}), 4)
        for child in children:
            self.assertGreaterEqual(child.box[0], root.box[0])
            self.assertGreaterEqual(child.box[1], root.box[1])
            self.assertLessEqual(child.box[2], root.box[2])
            self.assertLessEqual(child.box[3], root.box[3])
        grandchildren = generate_lazy_children(children[0])
        self.assertTrue(all(child.path[:1] == (0,) for child in grandchildren))

    def test_backtrack_chooses_best_unvisited_root_branch(self):
        roots = generate_lazy_children(AdaptivePatch((), (0, 0, 100, 100)))
        focus_child = generate_lazy_children(roots[0])[2]
        self.assertEqual(
            choose_backtrack_patch(roots, focus_child, visited={roots[0].identity}),
            roots[1],
        )


class LazyProjectionAndSelectionTests(unittest.TestCase):
    def test_two_scale_agreement_and_three_view_majority_project(self):
        agree = [
            {"winner": 1, "losses": [2.0, 1.0]},
            {"winner": 1, "losses": [3.0, 1.0]},
        ]
        projected = project_lazy_candidate("logits_match", ["x", "y"], agree)
        self.assertTrue(projected["feasible"])
        self.assertEqual(projected["output"], 1)
        self.assertEqual(projected["path_consensus"], 1.0)

        majority = [
            {"winner": 0, "losses": [1.0, 3.0]},
            {"winner": 1, "losses": [3.0, 1.0]},
            {"winner": 0, "losses": [1.0, 2.0]},
        ]
        projected = project_lazy_candidate("logits_match", ["x", "y"], majority)
        self.assertTrue(projected["feasible"])
        self.assertEqual(projected["output"], 0)
        self.assertEqual(projected["path_consensus"], 2 / 3)

    def test_rule_grid_is_fixed_and_selector_requires_all_three_margins(self):
        self.assertEqual(len(all_lazy_rule_keys()), 12)
        selected = select_lazy_candidate(
            p0(), candidate(), min_path_consensus=1.0,
            min_confidence=0.75, min_gain=0.25,
        )
        self.assertEqual(selected.action, "LAZY")
        retained = select_lazy_candidate(
            p0(), candidate(path_consensus=2 / 3), min_path_consensus=1.0,
            min_confidence=0.75, min_gain=0.25,
        )
        self.assertEqual(retained.action, "P0")


if __name__ == "__main__":
    unittest.main()
