import unittest

from cvsearch.perform_EGSearch import _validate_output


class TreeBenchRunnerContractTest(unittest.TestCase):
    def test_treebench_requires_a_string_output(self):
        policy = {
            "answer_type": "option_single",
            "options": "A. cat\nB. dog",
        }
        _validate_output("treebench", policy, "A")
        _validate_output("treebench", policy, "unparsed model text")
        with self.assertRaises(ValueError):
            _validate_output("treebench", policy, ["A"])


if __name__ == "__main__":
    unittest.main()
