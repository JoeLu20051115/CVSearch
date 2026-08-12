import math
import unittest
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
from PIL import Image

from cvsearch.evidence_gap.ranking import (
    ConservativeQueryRanker,
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

    def test_edge_density_streams_large_luminance_differences_without_concatenation(self):
        pixels = np.fromfunction(lambda y, x: (17 * x + 31 * y) % 256, (257, 513), dtype=int).astype(np.uint8)
        image = Image.fromarray(pixels, mode="L")
        signed = pixels.astype(np.int16)
        expected = (
            np.abs(signed[:, 1:] - signed[:, :-1]).sum(dtype=np.int64)
            + np.abs(signed[1:, :] - signed[:-1, :]).sum(dtype=np.int64)
        ) / ((pixels.shape[0] * (pixels.shape[1] - 1) + (pixels.shape[0] - 1) * pixels.shape[1]) * 255)
        with patch("cvsearch.evidence_gap.ranking.np.concatenate", side_effect=AssertionError("full differences must not concatenate")):
            self.assertAlmostEqual(edge_density(image), expected)

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
    def test_empty_candidates_bypass_query_validation_and_scorer(self):
        scorer = FakeScorer([[0.1]])
        ranker = QueryAwareNodeRanker(scorer)
        self.assertEqual(ranker.rank_with_details([], Image.new("RGB", (8, 8), "white"), None, None), ([], []))
        self.assertEqual(scorer.calls, [])

    def test_none_augmented_queries_means_main_only_for_nonempty_candidates(self):
        image = Image.new("RGB", (8, 8), "white")
        nodes = [FakeNode("low"), FakeNode("high")]
        scorer = FakeScorer([[0.1], [0.9]])
        ranked, details = QueryAwareNodeRanker(scorer).rank_with_details(nodes, image, "main", None)
        self.assertEqual([node.id for node in ranked], ["high", "low"])
        self.assertEqual(scorer.calls[0][1], ["main"])
        self.assertEqual([detail["score"]["augmented"] for detail in details], [1.0, 0.0])

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

    def test_query_demands_adapt_relevance_weight_and_are_traced(self):
        image = Image.new("RGB", (8, 8), "white")
        nodes = [FakeNode("relevant", complexity=0.0), FakeNode("informative", complexity=1.0)]
        detail_ranker = QueryAwareNodeRanker(
            FakeScorer([[1.0, 1.0], [0.0, 0.0]]),
            alpha=0.65,
            visual_lambda=1.0,
            detail_alpha_discount=0.65,
        )
        context_ranker = QueryAwareNodeRanker(
            FakeScorer([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
            alpha=0.25,
            visual_lambda=1.0,
            context_alpha_gain=0.75,
        )

        detail_nodes, detail = detail_ranker(
            nodes, image, "What color is the comb?", ["comb"]
        )
        context_nodes, context = context_ranker(
            nodes, image, "Is the comb left of the cup?", ["comb", "cup"]
        )

        self.assertEqual([node.id for node in detail_nodes], ["informative", "relevant"])
        self.assertEqual([node.id for node in context_nodes], ["relevant", "informative"])
        self.assertEqual(detail[0]["effective_alpha"], 0.0)
        self.assertEqual(detail[0]["query_profile"], {
            "detail_demand": 1.0, "context_demand": 0.0,
        })
        self.assertEqual(context[0]["effective_alpha"], 1.0)
        self.assertEqual(context[0]["query_profile"], {
            "detail_demand": 0.0, "context_demand": 1.0,
        })

    def test_context_demand_shifts_visual_fusion_from_complexity_to_edges(self):
        pixels = np.zeros((4, 8), dtype=np.uint8)
        pixels[:, 4:] = np.array([
            [0, 255, 0, 255],
            [255, 0, 255, 0],
            [0, 255, 0, 255],
            [255, 0, 255, 0],
        ], dtype=np.uint8)
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        nodes = [
            FakeNode("complex", (0, 0, 4, 4), complexity=1.0),
            FakeNode("edges", (4, 0, 4, 4), complexity=0.0),
        ]
        detail_ranker = QueryAwareNodeRanker(
            FakeScorer([[0.5, 0.5], [0.5, 0.5]]),
            alpha=0.0,
            visual_lambda=1.0,
            context_visual_discount=1.0,
        )
        context_ranker = QueryAwareNodeRanker(
            FakeScorer([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            alpha=0.0,
            visual_lambda=1.0,
            context_visual_discount=1.0,
        )

        detail_nodes, detail = detail_ranker(
            nodes, image, "What color is the comb?", ["comb"],
        )
        context_nodes, context = context_ranker(
            nodes, image, "Is the comb left of the cup?", ["comb", "cup"],
        )

        self.assertEqual([node.id for node in detail_nodes], ["complex", "edges"])
        self.assertEqual([node.id for node in context_nodes], ["edges", "complex"])
        self.assertEqual(detail[0]["effective_visual_lambda"], 1.0)
        self.assertEqual(context[0]["effective_visual_lambda"], 0.0)

    def test_attribute_qualified_context_retains_appearance_evidence(self):
        image = Image.new("RGB", (8, 4), "white")
        nodes = [
            FakeNode("a", (0, 0, 4, 4), complexity=1.0),
            FakeNode("b", (4, 0, 4, 4), complexity=0.0),
        ]
        ranker = QueryAwareNodeRanker(
            FakeScorer([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            alpha=0.25,
            visual_lambda=1.0,
            detail_alpha_discount=0.15,
            context_alpha_gain=0.45,
            context_visual_discount=1.0,
            appearance_descriptor_visual_relief=1.0,
        )

        _, details = ranker(
            nodes, image,
            "Is the green statue left of the white statue?",
            ["green statue", "white statue"],
        )

        self.assertEqual(details[0]["query_profile"], {
            "detail_demand": 0.0, "context_demand": 1.0,
            "appearance_demand": 1.0,
        })
        self.assertEqual(details[0]["effective_visual_lambda"], 1.0)
        self.assertAlmostEqual(details[0]["effective_alpha"], 0.7)

    def test_v5_soft_adaptation_keeps_context_weights_bounded(self):
        def details(question, augmented):
            ranker = QueryAwareNodeRanker(
                FakeScorer([[0.5] * (len(augmented) + 1)]),
                beta=0.6,
                alpha=0.6,
                visual_lambda=1.0,
                detail_alpha_discount=0.15,
                context_alpha_gain=0.15,
                context_visual_discount=0.5,
                appearance_descriptor_visual_relief=1.0,
            )
            return ranker(
                [FakeNode("node")], Image.new("RGB", (4, 4), "white"),
                question, augmented,
            )[1][0]

        detail = details("What is the color of the comb?", ["comb"])
        spatial = details("Is the comb left of the cup?", ["comb", "cup"])
        qualified = details(
            "Is the green statue left of the white statue?",
            ["green statue", "white statue"],
        )

        self.assertAlmostEqual(detail["effective_alpha"], 0.45)
        self.assertEqual(detail["effective_visual_lambda"], 1.0)
        self.assertAlmostEqual(spatial["effective_alpha"], 0.75)
        self.assertEqual(spatial["effective_visual_lambda"], 0.5)
        self.assertAlmostEqual(qualified["effective_alpha"], 0.75)
        self.assertEqual(qualified["effective_visual_lambda"], 1.0)

    def test_v1_omits_v2_visual_trace_field(self):
        _, details = QueryAwareNodeRanker(FakeScorer([[0.5]]))(
            [FakeNode("node")], Image.new("RGB", (4, 4), "white"), "main", [],
        )
        self.assertNotIn("effective_visual_lambda", details[0])

    def test_adaptive_weights_are_validated(self):
        for detail_discount, context_gain, visual_discount in (
            (True, 0.0, None),
            (-0.1, 0.0, None),
            (0.0, 1.1, None),
            (0.0, 0.0, True),
            (0.0, 0.0, 1.1),
        ):
            with self.subTest(
                detail_discount=detail_discount, context_gain=context_gain,
                visual_discount=visual_discount,
            ):
                with self.assertRaises((TypeError, ValueError)):
                    QueryAwareNodeRanker(
                        FakeScorer([[0.1]]),
                        detail_alpha_discount=detail_discount,
                        context_alpha_gain=context_gain,
                        context_visual_discount=visual_discount,
                    )

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


class ConservativeQueryRankerTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (8, 8), "white")
        self.nodes = [FakeNode("a"), FakeNode("b"), FakeNode("c"), FakeNode("d")]

    @staticmethod
    def _reversing_ranker():
        def ranker(nodes, image_pil, main_query, augmented_queries):
            ranked = list(reversed(nodes))
            return ranked, [{"node_id": node.id} for node in ranked]
        return ranker

    @staticmethod
    def _tied_ranker():
        def ranker(nodes, image_pil, main_query, augmented_queries):
            ranked = list(nodes)
            return ranked, [{"node_id": node.id} for node in ranked]
        return ranker

    def test_conservative_ranker_preserves_identity_and_max_displacement(self):
        ranked, details = ConservativeQueryRanker(
            self._reversing_ranker(), rho=0.25, max_displacement=1
        )(self.nodes, self.image, "question", ["evidence"])
        self.assertCountEqual(map(id, ranked), map(id, self.nodes))
        original = {id(node): index for index, node in enumerate(self.nodes)}
        self.assertTrue(all(abs(index - original[id(node)]) <= 1
                            for index, node in enumerate(ranked)))
        self.assertEqual([detail["node_id"] for detail in details], [node.id for node in ranked])

    def test_zero_rho_is_exact_cvsearch_order(self):
        ranked, details = ConservativeQueryRanker(
            self._reversing_ranker(), rho=0.0, max_displacement=4
        )(self.nodes, self.image, "question", ["evidence"])
        self.assertEqual(ranked, self.nodes)
        self.assertEqual([detail["node_id"] for detail in details], [node.id for node in self.nodes])

    def test_ties_keep_original_order_and_details_record_both_ranks(self):
        ranked, details = ConservativeQueryRanker(
            self._tied_ranker(), rho=0.5, max_displacement=2
        )(self.nodes, self.image, "question", ["evidence"])
        self.assertEqual(ranked, self.nodes)
        self.assertIn("cvsearch_rank", details[0])
        self.assertIn("query_rank", details[0])
        self.assertIn("fused_rank_score", details[0])

    def test_accepts_mapping_details_from_the_base_ranker(self):
        def mapping_ranker(nodes, image_pil, main_query, augmented_queries):
            return list(nodes), [MappingProxyType({"node_id": node.id}) for node in nodes]

        ranked, _ = ConservativeQueryRanker(mapping_ranker, rho=0.5, max_displacement=1)(
            self.nodes, self.image, "question", ["evidence"]
        )
        self.assertEqual(ranked, self.nodes)

    def test_rejects_invalid_parameters_duplicate_nodes_and_misaligned_details(self):
        for rho, displacement in ((True, 1), (math.nan, 1), (-0.1, 1), (0.5, -1), (0.5, True)):
            with self.subTest(rho=rho, displacement=displacement):
                with self.assertRaises((TypeError, ValueError)):
                    ConservativeQueryRanker(self._tied_ranker(), rho, displacement)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            ConservativeQueryRanker(self._tied_ranker(), 0.5, 1)(
                [self.nodes[0], self.nodes[0]], self.image, "question", ["evidence"]
            )

        def bad_details(nodes, image_pil, main_query, augmented_queries):
            return list(reversed(nodes)), [{"node_id": node.id} for node in nodes]

        with self.assertRaisesRegex(ValueError, "details"):
            ConservativeQueryRanker(bad_details, 0.5, 1)(
                self.nodes, self.image, "question", ["evidence"]
            )


if __name__ == "__main__":
    unittest.main()
