import unittest

from cvsearch.eval.candidate_free_verifier_selector import (
    CandidateFreeRuntimeEvidence,
    CandidateFreeVerifierEvidence,
    candidate_free_cascade_outcome,
    candidate_free_runtime_outcome,
)
from cvsearch.eval.freeze_uncertainty_support import _RiskExample, _RiskTopic
from cvsearch.eval.replay_uncertainty_support import (
    AdvantageFeatures,
    AggregateEvidenceFeatures,
    CandidateSnapshot,
    RiskCalibrator,
    RiskLinearHead,
    RiskLogisticHead,
)
from tests.test_freeze_uncertainty_support import record


def constant_head(probability):
    import math

    width = len(AggregateEvidenceFeatures.__dataclass_fields__)
    return RiskLogisticHead(
        intercept=math.log(probability / (1.0 - probability)),
        coefficients=(0.0,) * width,
        means=(0.0,) * width,
        scales=(1.0,) * width,
    )


def topic():
    example = _RiskExample(
        group="group",
        features=AdvantageFeatures(0.8, 0.7, 0.6, 0.1, 0.2),
        evidence_features=AggregateEvidenceFeatures(
            0.8, 0.7, 0.6, 0.1, 0.2, 0.6, 0.1,
            0.2, 0.8, 1.0, 0.5, 0.2, 0.1, 0.2,
        ),
        checkpoint=(0, ("tight", "context")),
        observations=2,
        agreeing_views=2,
        candidate_canonical="bad",
        correction_units=0,
        corruption_units=1,
        official_units=1,
    )
    return _RiskTopic(
        record=record("group", "qwen", helpful=False, ordinal=0),
        examples=(example,),
        stop_observations=8,
    )


def calibrator():
    zero = RiskLinearHead((0.0,) * 18)
    return RiskCalibrator(
        zero, zero, 1.0, 0.0,
        evidence_benefit_head=constant_head(0.9),
        evidence_harm_head=constant_head(0.1),
        minimum_observations=1,
        maximum_observations=8,
        minimum_agreeing_views=2,
    )


def snapshot(output="candidate"):
    features = AdvantageFeatures(0.8, 0.7, 0.6, 0.1, 0.2)
    return CandidateSnapshot(
        branch_index=0,
        revealed_roles=("tight", "context"),
        observations=8,
        output=output,
        canonical_answer=output,
        agreeing_hashes=("a", "b"),
        parseable_count=2,
        raw_support_floor=0.0,
        features=features,
        evidence_features=AggregateEvidenceFeatures(
            0.8, 0.7, 0.6, 0.1, 0.2, 0.6, 0.1,
            0.2, 0.8, 1.0, 0.5, 0.2, 0.1, 0.2,
        ),
        raw_score=0.0,
        calibrated_advantage=0.0,
        structurally_eligible=True,
    )


class CandidateFreeCascadeTests(unittest.TestCase):
    def test_runtime_cascade_vetoes_at_point_six_but_proposal_requires_point_nine(self):
        evidence = CandidateFreeRuntimeEvidence(
            verifier_feasible=True,
            verifier_canonical="proposal",
            verifier_confidence=0.7,
            proposal_feasible=True,
            proposal_output="proposal",
            proposal_canonical="proposal",
            proposal_agreement=0.8,
            observations=10,
        )

        outcome = candidate_free_runtime_outcome(
            (snapshot(),), "p0", calibrator(), evidence,
            veto_confidence=0.6,
            proposal_confidence=0.9,
            proposal_agreement=0.4,
        )

        self.assertEqual(outcome.selected_output, "p0")
        self.assertEqual(outcome.selected_source, "P0")
        self.assertTrue(outcome.vetoed)

    def test_runtime_cascade_accepts_exact_high_confidence_proposal_after_veto(self):
        evidence = CandidateFreeRuntimeEvidence(
            verifier_feasible=True,
            verifier_canonical="proposal",
            verifier_confidence=0.95,
            proposal_feasible=True,
            proposal_output="proposal raw",
            proposal_canonical="proposal",
            proposal_agreement=0.4,
            observations=10,
        )

        outcome = candidate_free_runtime_outcome(
            (snapshot(),), "p0", calibrator(), evidence,
            veto_confidence=0.6,
            proposal_confidence=0.9,
            proposal_agreement=0.4,
        )

        self.assertEqual(outcome.selected_output, "proposal raw")
        self.assertEqual(outcome.selected_source, "VERIFIED_PROPOSAL")
        self.assertTrue(outcome.vetoed)

    def test_high_confidence_disagreement_vetoes_harm_then_uses_safe_proposal(self):
        evidence = CandidateFreeVerifierEvidence(
            verifier_feasible=True,
            verifier_canonical="good",
            verifier_confidence=0.9,
            proposal_feasible=True,
            proposal_canonical="good",
            proposal_agreement=0.7,
            proposal_corrections=2,
            proposal_corruptions=0,
            observations=10,
        )

        outcome = candidate_free_cascade_outcome(
            topic(), calibrator(), evidence,
        )

        self.assertEqual(outcome.net_gain, 2)
        self.assertEqual(outcome.corrections, 2)
        self.assertEqual(outcome.corruptions, 0)
        self.assertTrue(outcome.vetoed)
        self.assertEqual(outcome.selected_source, "VERIFIED_PROPOSAL")
        self.assertEqual(outcome.observations, 10)

    def test_low_confidence_disagreement_cannot_change_aggregate_decision(self):
        evidence = CandidateFreeVerifierEvidence(
            verifier_feasible=True,
            verifier_canonical="good",
            verifier_confidence=0.59,
            proposal_feasible=True,
            proposal_canonical="good",
            proposal_agreement=1.0,
            proposal_corrections=2,
            proposal_corruptions=0,
            observations=10,
        )

        outcome = candidate_free_cascade_outcome(
            topic(), calibrator(), evidence,
        )

        self.assertEqual(outcome.net_gain, -1)
        self.assertFalse(outcome.vetoed)
        self.assertEqual(outcome.selected_source, "AGGREGATE")


if __name__ == "__main__":
    unittest.main()
