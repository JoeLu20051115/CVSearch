import ast
import inspect
import unittest
from pathlib import Path

from cvsearch.evidence_gap.baselines import baseline_envelope

ROOT = Path(__file__).resolve().parents[1]


def parse(relative_path):
    return ast.parse((ROOT / relative_path).read_text())


class PaperParameterTest(unittest.TestCase):
    def test_qwen_search_thresholds_match_paper_configuration(self):
        tree = parse("cvsearch/perform_CVSearch.py")
        qwen_kwargs = None

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "search_kwargs" for target in node.targets):
                continue
            if not isinstance(node.value, ast.Dict):
                continue

            values = {}
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant):
                    try:
                        values[key.value] = ast.literal_eval(value)
                    except ValueError:
                        pass
            if values.get("threshold_descrease") == [0.05, 0.1, 0.2]:
                qwen_kwargs = values
                break

        self.assertIsNotNone(qwen_kwargs)
        self.assertEqual(qwen_kwargs["answering_confidence_threshold_lower"], 0)
        self.assertEqual(qwen_kwargs["answering_confidence_threshold_upper"], 0.9)
        self.assertEqual(qwen_kwargs["fast_threshold"], 0.6)

    def test_dual_baseline_envelope_retains_both_gate_defaults(self):
        default = inspect.signature(baseline_envelope).parameters["thresholds"].default
        self.assertEqual(default, (0.6, 0.8))

    def test_tree_and_ranking_parameters_match_paper(self):
        tree = parse("cvsearch/CVSearch.py")
        assignments = {}
        split_ranges = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value.value
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "build_tree":
                keywords = {
                    keyword.arg: ast.literal_eval(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg in {"min_splits", "max_splits"}
                }
                split_ranges.append((keywords["min_splits"], keywords["max_splits"]))

        self.assertEqual(assignments["tree_prune_threshold"], 0.4)
        self.assertEqual(assignments["tree_depth_s"], 2)
        self.assertEqual(assignments["tree_depth_c"], 3)
        self.assertTrue(split_ranges)
        self.assertTrue(all(split_range == (4, 8) for split_range in split_ranges))

        search = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "semantic_guide_search_dynamic_depth")
        defaults = {
            argument.arg: ast.literal_eval(default)
            for argument, default in zip(search.args.args[-len(search.args.defaults):], search.args.defaults)
        }
        self.assertEqual(defaults["w_prior"], 0.2)
        self.assertEqual(defaults["w_current"], 0.4)
        self.assertEqual(defaults["w_child"], 0.4)


if __name__ == "__main__":
    unittest.main()
