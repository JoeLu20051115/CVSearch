import copy
import unittest

from PIL import Image

from cvsearch.eval.pairwise_verifier import (
    IndependentAnswerProjection,
    PairwiseProjection,
    compose_independent_source_view,
    compose_pairwise_evidence_sheet,
    independent_answer_decision,
    independent_answer_prompt_material,
    pairwise_answer_display,
    pairwise_decision,
    pairwise_prompt_material,
    project_independent_answer,
    project_pairwise_losses,
    propose_pairwise_candidate,
    select_pairwise_evidence_views,
    source_pairwise_prompt_material,
)
from tests.test_replay_split_search import calibration, rescue_rows


def split_audit(row):
    return row["method_trace"]["steps"][0]["split_search_audit"]


class PairwiseProposalTests(unittest.TestCase):
    def test_proposal_is_label_blind_and_requires_two_agreeing_views(self):
        stage2, split = rescue_rows()
        branches = split_audit(split)["branches"]
        for branch in branches:
            for role in ("tight_view", "medium_view", "context_view"):
                if role in branch:
                    branch[role]["answer"] = "A"
        for branch in branches[:2]:
            for role in ("tight_view", "context_view"):
                branch[role]["answer"] = "B"
                branch[role]["raw_support"] = 0.9

        first = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )
        stage2["answer"] = "B"
        stage2["category"] = "poison"
        split["answer"] = "C"
        split["category"] = "different"
        second = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )

        self.assertEqual(first, second)
        self.assertTrue(first.feasible)
        self.assertEqual(first.candidate_output, "B")
        self.assertGreaterEqual(len(first.agreeing_hashes), 2)
        self.assertLessEqual(first.observations, 8)

    def test_proposal_falls_back_when_confirmation_is_insufficient(self):
        stage2, split = rescue_rows()
        branches = split_audit(split)["branches"]
        for branch in branches:
            for role in ("tight_view", "medium_view", "context_view"):
                if role in branch:
                    branch[role]["answer"] = "A"
        branches[0]["tight_view"]["answer"] = "B"

        proposal = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )

        self.assertFalse(proposal.feasible)
        self.assertEqual(proposal.stage2_output, stage2["output"])

    def test_proposal_uses_latest_checkpoint_that_still_meets_agreement(self):
        stage2, split = rescue_rows()
        branches = split_audit(split)["branches"]
        for branch in branches:
            for role in ("tight_view", "medium_view", "context_view"):
                if role in branch:
                    branch[role]["answer"] = "A"
        for branch, role in (
            (branches[0], "tight_view"),
            (branches[0], "context_view"),
            (branches[1], "tight_view"),
        ):
            branch[role]["answer"] = "B"

        proposal = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )

        self.assertTrue(proposal.feasible)
        self.assertEqual(proposal.candidate_output, "B")
        self.assertEqual(proposal.observations, 7)


class PairwiseProjectionTests(unittest.TestCase):
    def test_independent_answer_prompt_never_exposes_competing_answers(self):
        material = independent_answer_prompt_material(
            "option_single",
            "Which item is closest?",
            "A. Tree\nB. Bench\nC. Road\nD. Sign\nE. Car\nF. House",
        )

        self.assertEqual(
            material["choices"], [["A", "B", "C", "D", "E", "F"]],
        )
        self.assertEqual(len(material["prompts"]), 1)
        self.assertIn("A. Tree", material["prompts"][0])
        self.assertIn("F. House", material["prompts"][0])
        for forbidden in ("proposal", "candidate", "p0", "correctness"):
            self.assertNotIn(forbidden, repr(material).lower())

    def test_independent_hr_answer_projects_semantic_shuffle_consensus(self):
        options = [
            "A. red\nB. blue\nC. green\nD. black",
            "A. green\nB. black\nC. blue\nD. red",
            "A. black\nB. red\nC. blue\nD. green",
            "A. blue\nB. green\nC. red\nD. black",
        ]
        observations = [
            {"winner": winner, "losses": losses}
            for winner, losses in (
                (1, [2.0, 0.0, 3.0, 4.0]),
                (2, [2.0, 3.0, 0.0, 4.0]),
                (2, [2.0, 3.0, 0.0, 4.0]),
                (0, [0.0, 2.0, 3.0, 4.0]),
            )
        ]

        projection = project_independent_answer(
            "option_list", options, observations,
        )

        self.assertTrue(projection.feasible)
        self.assertEqual(projection.canonical_answer, "blue")
        self.assertEqual(projection.output, ["B", "C", "C", "A"])
        self.assertGreater(projection.confidence, 0.7)

    def test_independent_answer_requires_exact_canonical_agreement(self):
        stage2, split = rescue_rows()
        branches = split_audit(split)["branches"]
        for branch in branches[:2]:
            for role in ("tight_view", "context_view"):
                branch[role]["answer"] = "B"
                branch[role]["raw_support"] = 0.9
        proposal = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )

        rejected = independent_answer_decision(
            proposal,
            IndependentAnswerProjection(True, "C", "C", 0.99),
            agreement_threshold=0.4,
            confidence_threshold=0.5,
            verifier_calls=1,
        )

        self.assertEqual(rejected["selected_source"], "P0")
        self.assertEqual(rejected["selected_output"], stage2["output"])
        self.assertEqual(rejected["observations"], proposal.observations + 1)

    def test_answer_display_resolves_index_and_letter_without_labels(self):
        self.assertEqual(
            pairwise_answer_display(["black", "red", "white"], 2),
            "white",
        )
        self.assertEqual(
            pairwise_answer_display("A. Feet\nB. Waist\nC. Head", "B"),
            "B. Waist",
        )
        self.assertEqual(pairwise_answer_display([], "blue"), "blue")

    def test_prompt_reverses_answer_order_without_hidden_metadata(self):
        material = pairwise_prompt_material(
            "What color is the sign?", ["red", "blue"], "red", "blue",
        )

        self.assertEqual(material["candidate_choice_indices"], [1, 0])
        self.assertIn('Answer 1: "red"', material["prompts"][0])
        self.assertIn('Answer 2: "blue"', material["prompts"][0])
        self.assertIn('Answer 1: "blue"', material["prompts"][1])
        self.assertIn('Answer 2: "red"', material["prompts"][1])
        serialized = repr(material).lower()
        for forbidden in ("correctness", "category", "dataset", "backbone"):
            self.assertNotIn(forbidden, serialized)

    def test_source_prompt_uses_short_order_reversed_choices(self):
        material = source_pairwise_prompt_material(
            "What color is visible?", "red", "blue",
        )

        self.assertEqual(material["choices"], ["1", "2"])
        self.assertEqual(material["candidate_choice_indices"], [1, 0])
        self.assertIn("Proposal 1: red", material["prompts"][0])
        self.assertIn("Proposal 2: blue", material["prompts"][0])
        self.assertIn("Proposal 1: blue", material["prompts"][1])
        self.assertIn("Proposal 2: red", material["prompts"][1])

    def test_reversed_loss_votes_project_to_shared_candidate_probability(self):
        projection = project_pairwise_losses((
            {"winner": 1, "losses": [1.0, 0.0]},
            {"winner": 0, "losses": [0.2, 1.2]},
        ))

        self.assertTrue(projection.feasible)
        self.assertGreater(projection.confidence, 0.7)
        self.assertEqual(len(projection.candidate_probabilities), 2)

    def test_order_disagreement_is_infeasible_even_with_one_strong_vote(self):
        projection = project_pairwise_losses((
            {"winner": 1, "losses": [4.0, 0.0]},
            {"winner": 1, "losses": [4.0, 0.0]},
        ))

        self.assertFalse(projection.feasible)

    def test_decision_counts_both_verifier_calls_and_falls_back_exactly(self):
        stage2, split = rescue_rows()
        branches = split_audit(split)["branches"]
        for branch in branches[:2]:
            for role in ("tight_view", "context_view"):
                branch[role]["answer"] = "B"
                branch[role]["raw_support"] = 0.9
        proposal = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )
        rejected = pairwise_decision(
            proposal,
            PairwiseProjection(False, 0.9, (0.9, 0.9)),
            agreement_threshold=0.4,
            confidence_threshold=0.5,
        )

        self.assertEqual(rejected["selected_source"], "P0")
        self.assertEqual(rejected["selected_output"], stage2["output"])
        self.assertEqual(rejected["observations"], proposal.observations + 2)


class PairwiseEvidenceTests(unittest.TestCase):
    def test_independent_source_view_is_an_exact_nonmutating_copy(self):
        source = Image.new("RGB", (117, 83), (11, 22, 33))

        view, audit = compose_independent_source_view(source)

        self.assertIsNot(view, source)
        self.assertEqual(view.tobytes(), source.tobytes())
        self.assertEqual(audit["view_size"], [117, 83])
        self.assertEqual(len(audit["view_sha256"]), 64)

    def test_selects_two_strongest_agreeing_views_and_renders_deterministically(self):
        stage2, split = rescue_rows()
        branches = split_audit(split)["branches"]
        for branch_index, branch in enumerate(branches[:2]):
            for role_index, role in enumerate(("tight_view", "context_view")):
                branch[role]["answer"] = "B"
                branch[role]["raw_support"] = (
                    0.5 + 0.1 * branch_index + 0.01 * role_index
                )
        proposal = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )

        views = select_pairwise_evidence_views(split, proposal)
        source = Image.new("RGB", (100, 100), (20, 40, 60))
        original = source.tobytes()
        first, first_audit = compose_pairwise_evidence_sheet(source, views)
        second, second_audit = compose_pairwise_evidence_sheet(source, views)

        self.assertEqual(len(views), 2)
        self.assertGreaterEqual(views[0].raw_support, views[1].raw_support)
        self.assertTrue(set(view.render_sha256 for view in views).issubset(
            proposal.agreeing_hashes,
        ))
        self.assertEqual(first.size, (896, 896))
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(source.tobytes(), original)

    def test_rejects_evidence_when_source_size_drifted(self):
        stage2, split = rescue_rows()
        branches = split_audit(split)["branches"]
        for branch in branches[:2]:
            for role in ("tight_view", "context_view"):
                branch[role]["answer"] = "B"
                branch[role]["raw_support"] = 0.9
        proposal = propose_pairwise_candidate(
            stage2, split, calibration(), minimum_agreement=0.4,
        )
        views = select_pairwise_evidence_views(split, proposal)

        with self.assertRaisesRegex(ValueError, "source size"):
            compose_pairwise_evidence_sheet(
                Image.new("RGB", (101, 100)), views,
            )


if __name__ == "__main__":
    unittest.main()
