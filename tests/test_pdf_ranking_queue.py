import math
import unittest

from cvsearch.evidence_gap.pdf_controller import FrozenRankedQueue
from cvsearch.evidence_gap.pdf_types import CandidateDescriptor


def candidate(key, sibling, ordinal):
    return CandidateDescriptor(
        canonical_key=key,
        sibling_group=sibling,
        native_ordinal=ordinal,
        bbox_original=(float(ordinal), 0.0, 10.0, 10.0),
        depth=1,
        render_level=0,
    )


class FrozenRankedQueueTest(unittest.TestCase):
    def test_preserves_all_candidates_and_pops_direct_joint_rank(self):
        queue = FrozenRankedQueue()
        native = [candidate("a", "root", 0), candidate("b", "root", 1), candidate("c", "root", 2)]
        queue.add_sibling_group(
            native,
            ranked_keys=("c", "b", "a"),
            score_by_key={"a": 0.1, "b": 0.6, "c": 0.9},
        )

        self.assertEqual(queue.candidate_count, 3)
        self.assertEqual(queue.native_first_choice_changes, 1)
        self.assertEqual([queue.pop_next().canonical_key for _ in range(3)], ["c", "b", "a"])
        self.assertIsNone(queue.pop_next())
        self.assertEqual(queue.popped_scores, (0.9, 0.6, 0.1))

    def test_global_queue_uses_frozen_scores_and_stable_discovery_ties(self):
        queue = FrozenRankedQueue()
        queue.add_sibling_group(
            [candidate("a", "g1", 0), candidate("b", "g1", 1)],
            ranked_keys=("b", "a"), score_by_key={"a": 0.2, "b": 0.8},
        )
        queue.add_sibling_group(
            [candidate("c", "g2", 0), candidate("d", "g2", 1)],
            ranked_keys=("c", "d"), score_by_key={"c": 0.8, "d": 0.3},
        )
        self.assertEqual([queue.pop_next().canonical_key for _ in range(4)], ["b", "c", "d", "a"])

    def test_rejects_pruning_duplicates_misalignment_and_nonfinite_scores(self):
        a, b = candidate("a", "root", 0), candidate("b", "root", 1)
        bad_calls = (
            ((a, b), ("a",), {"a": 0.5, "b": 0.4}),
            ((a, b), ("a", "a"), {"a": 0.5, "b": 0.4}),
            ((a, b), ("a", "b"), {"a": 0.5}),
            ((a, b), ("a", "b"), {"a": math.nan, "b": 0.4}),
        )
        for native, ranked, scores in bad_calls:
            with self.subTest(ranked=ranked, scores=scores), self.assertRaises((TypeError, ValueError)):
                FrozenRankedQueue().add_sibling_group(native, ranked, scores)

        queue = FrozenRankedQueue()
        queue.add_sibling_group((a, b), ("a", "b"), {"a": 0.5, "b": 0.4})
        with self.assertRaisesRegex(ValueError, "already"):
            queue.add_sibling_group((a,), ("a",), {"a": 0.5})


if __name__ == "__main__":
    unittest.main()
