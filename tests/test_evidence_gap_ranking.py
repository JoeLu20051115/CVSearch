import math
import unittest

import numpy as np
from PIL import Image

from cvsearch.evidence_gap.ranking import (
    QueryAwareNodeRanker,
    edge_density,
    fuse_scores,
    percentile,
)


class FakeState:
    def __init__(self, bbox):
        self.bbox = bbox


class FakeNode:
    def __init__(self, identifier, bbox=(0, 0, 4, 4), complexity=0.0):
        self.id = identifier
        self.state = FakeState(bbox)
        self.complexity = complexity


class FakeScorer:
    def __init__(self, matrix):
        self.matrix = matrix
        self.calls = []

    def score(self, images, texts):
        self.calls.append((list(images), list(texts)))
        return self.matrix


class RankingUtilitiesTest(unittest.TestCase):
    def test_percentile_uses_average_tie_ranks_and_preserves_input(self):
        values = [3.0, 1.0, 1.0, 2.0]
        self.assertEqual(percentile(values), [1.0, 1 / 6, 1 / 6, 2 / 3])
        self.assertEqual(values, [3.0, 1.0, 1.0, 2.0])

    def test_percentile_uses_neutral_value_for_singleton_and_constants(self):
        self.assertEqual(percentile([7.0]), [0.5])
        self.assertEqual(percentile([7.0, 7.0, 7.0]), [0.5, 0.5, 0.5])
        self.assertEqual(percentile([]), [])

    def test_percentile_rejects_booleans_and_nonfinite_values_but_accepts_numpy_scalars(self):
        self.assertEqual(percentile([np.float32(0.0), np.float64(1.0)]), [0.0, 1.0])
        for values in ([True], [math.nan], [math.inf], ["1"]):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    percentile(values)

    def test_edge_density_is_bounded_and_does_not_change_images(self):
        constant = Image.new("RGB", (3, 3), (13, 13, 13))
        checker = Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8), mode="L")
        before = constant.tobytes()
        self.assertEqual(edge_density(Image.new("L", (1, 1), 0)), 0.0)
        self.assertEqual(edge_density(constant), 0.0)
        self.assertGreater(edge_density(checker), 0.99)
        self.assertLessEqual(edge_density(checker), 1.0)
        self.assertEqual(constant.tobytes(), before)

    def test_fuse_scores_percentiles_every_component_and_returns_frozen_scores(self):
        main = [0.0, 1.0]
        augmented = [1.0, 0.0]
        complexity = [3.0, 3.0]
        edge = [0.0, 1.0]
        scores = fuse_scores(main, augmented, complexity, edge, beta=0.5, alpha=0.5, visual_lambda=0.5)
        self.assertEqual([score.main for score in scores], [0.0, 1.0])
        self.assertEqual([score.augmented for score in scores], [1.0, 0.0])
        self.assertEqual([score.complexity for score in scores], [0.5, 0.5])
        self.assertEqual([score.edge_density for score in scores], [0.0, 1.0])
        self.assertEqual([score.rank for score in scores], [0.375, 0.625])
        self.assertTrue(all(0.0 <= value <= 1.0 for score in scores for value in score.to_dict().values()))
        self.assertEqual(main, [0.0, 1.0])
        self.assertEqual(fuse_scores([], [], [], [], 0.5, 0.5, 0.5), [])

    def test_fuse_scores_rejects_invalid_weights_lengths_and_components(self):
        for args in (
            ([0.0], [], [0.0], [0.0], 0.5, 0.5, 0.5),
            ([0.0], [0.0], [0.0], [0.0], True, 0.5, 0.5),
            ([0.0], [0.0], [0.0], [0.0], 1.1, 0.5, 0.5),
            ([math.nan], [0.0], [0.0], [0.0], 0.5, 0.5, 0.5),
        ):
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError)):
                    fuse_scores(*args)


class QueryAwareNodeRankerTest(unittest.TestCase):
    def test_all_candidates_survive_and_ties_are_equal(self):
        image = Image.new("RGB", (8, 8), "white")
        nodes = [FakeNode("a"), FakeNode("b"), FakeNode("c")]
        scorer = FakeScorer([[0.2, 0.1], [0.2, 0.1], [0.2, 0.1]])
        ranker = QueryAwareNodeRanker(scorer)

        ranked, details = ranker.rank_with_details(nodes, image, "bus sign", ["blue road sign"])

        self.assertEqual([node.id for node in ranked], ["a", "b", "c"])
        self.assertEqual(len(details), 3)
        self.assertEqual([detail["node_id"] for detail in details], ["a", "b", "c"])
        self.assertTrue(all(detail["score"]["rank"] == 0.5 for detail in details))
        self.assertEqual([node.id for node in nodes], ["a", "b", "c"])
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(scorer.calls[0][1], ["bus sign", "blue road sign"])

    def test_ranker_uses_main_when_there_are_no_augmented_queries(self):
        image = Image.new("RGB", (8, 8), "white")
        nodes = [FakeNode("low"), FakeNode("high")]
        scorer = FakeScorer([[0.1], [0.9]])
        ranked, details = QueryAwareNodeRanker(scorer)(nodes, image, "main", [])

        self.assertEqual([node.id for node in ranked], ["high", "low"])
        self.assertEqual([detail["score"]["augmented"] for detail in details], [1.0, 0.0])

    def test_ranker_uses_top_three_augmented_scores_and_clamps_crops(self):
        image = Image.new("RGB", (8, 8), "white")
        nodes = [FakeNode("outside", (-2, -1, 5, 5), 1.0), FakeNode("inside", (2, 2, 3, 3), 0.0)]
        scorer = FakeScorer([[0.3, 0.0, 0.4, 0.6, 1.0], [0.3, 0.1, 0.2, 0.3, 0.4]])
        ranked, details = QueryAwareNodeRanker(scorer)(nodes, image, "main", ["a", "b", "c", "d"])

        self.assertEqual([node.id for node in ranked], ["outside", "inside"])
        self.assertAlmostEqual(details[0]["score"]["augmented"], 1.0)
        self.assertAlmostEqual(details[1]["score"]["augmented"], 0.0)
        self.assertEqual(scorer.calls[0][0][0].size, (3, 4))

    def test_ranker_rejects_invalid_bbox_and_invalid_scorer_matrices(self):
        image = Image.new("RGB", (8, 8), "white")
        for node in (FakeNode("bad-width", (0, 0, 0, 1)), FakeNode("outside", (20, 0, 1, 1)), FakeNode("nan", (0, 0, math.nan, 1))):
            with self.subTest(node=node.id):
                with self.assertRaises((TypeError, ValueError)):
                    QueryAwareNodeRanker(FakeScorer([[0.1]])).rank([node], image, "main", [])
        for matrix in ([[0.1, 0.2]], [[math.nan]], [[0.1], [0.2]]):
            with self.subTest(matrix=matrix):
                with self.assertRaises((TypeError, ValueError)):
                    QueryAwareNodeRanker(FakeScorer(matrix)).rank([FakeNode("a")], image, "main", [])


if __name__ == "__main__":
    unittest.main()
