import json
import unittest

from cvsearch.eval.freeze_uncertainty_support import (
    DevelopmentRecord,
    NoFeasibleConfiguration,
    canonical_payload_hash,
    freeze_policy,
    select_configuration,
    source_group,
    utility_target,
)
from tests.test_replay_split_search import calibration, rescue_rows


def record(
    group,
    backbone,
    *,
    helpful,
    ordinal=0,
):
    phase1, split = rescue_rows()
    for branch in split["method_trace"]["steps"][0][
        "split_search_audit"
    ]["branches"]:
        for role in ("tight_view", "medium_view", "context_view"):
            if role in branch:
                branch[role]["answer"] = "A"
                branch[role]["raw_support"] = 0.9
    first = split["method_trace"]["steps"][0]["split_search_audit"][
        "branches"
    ][0]
    first["tight_view"]["answer"] = "B"
    first["context_view"]["answer"] = "B"
    for row in (phase1, split):
        row["_eg_ordinal"] = ordinal
        row["input_image"] = f"images/{group}.jpg"
        row["answer"] = "B" if helpful else "A"
    return DevelopmentRecord(
        group=f"treebench:images/{group}.jpg",
        backbone=backbone,
        benchmark="treebench",
        ordinal=ordinal,
        stage2_row=phase1,
        split_row=split,
        calibration=calibration(),
    )


def helpful_topics():
    return (
        record("g1", "qwen", helpful=True),
        record("g2", "internvl", helpful=True, ordinal=1),
    )


class SourceGroupingTests(unittest.TestCase):
    def test_hr_resolutions_and_backbones_share_one_group(self):
        self.assertEqual(
            source_group("hr_bench_4k", 7, "4k.png"),
            source_group("hr_bench_8k", 7, "8k.png"),
        )

    def test_non_hr_uses_normalized_source_path(self):
        self.assertEqual(
            source_group("vstar", 2, "./images/x.jpg"),
            "vstar:images/x.jpg",
        )

    def test_source_group_rejects_invalid_identity(self):
        for values in (
            ("hr_bench_4k", -1, "x"),
            ("vstar", 0, ""),
            ("unknown", 0, "x"),
        ):
            with self.subTest(values=values):
                with self.assertRaises((TypeError, ValueError)):
                    source_group(*values)


class UtilityTargetTests(unittest.TestCase):
    def test_utility_target_maps_harm_neutral_help_around_half(self):
        self.assertEqual(utility_target(-1, 2), 0.25)
        self.assertEqual(utility_target(0, 2), 0.5)
        self.assertEqual(utility_target(1, 2), 0.75)

    def test_utility_target_rejects_impossible_delta(self):
        with self.assertRaises(ValueError):
            utility_target(2, 1)


class GroupedSelectionTests(unittest.TestCase):
    def test_oof_calibrator_never_sees_held_out_group(self):
        selection = select_configuration(helpful_topics())

        self.assertEqual(len(selection.folds), 2)
        for fold in selection.folds:
            self.assertTrue(
                set(fold.train_groups).isdisjoint(fold.held_out_groups),
            )

    def test_tie_breaks_by_declared_profile_and_threshold_order(self):
        selection = select_configuration(helpful_topics())

        self.assertEqual(
            (selection.profile, selection.threshold),
            ("balanced", 0.0),
        )
        self.assertEqual(selection.metrics.net_gain, 2)
        self.assertEqual(selection.metrics.corruptions, 0)

    def test_configuration_requires_no_harm_in_every_available_cell(self):
        topics = (
            record("g1", "qwen", helpful=True),
            record("g2", "internvl", helpful=False, ordinal=1),
        )

        with self.assertRaises(NoFeasibleConfiguration):
            select_configuration(topics)

    def test_frozen_payload_is_deterministic_and_contains_no_labels(self):
        provenance = {"development_sha256": "a" * 64}

        first = freeze_policy(helpful_topics(), provenance=provenance)
        second = freeze_policy(helpful_topics(), provenance=provenance)

        self.assertEqual(first, second)
        self.assertEqual(first["payload_sha256"], canonical_payload_hash(first))
        serialized = json.dumps(first, sort_keys=True)
        self.assertNotIn('"answer"', serialized)
        self.assertNotIn('"correct"', serialized)
        self.assertNotIn('"utility_target"', serialized)
        self.assertEqual(first["data_scope"], "opened_development_only")

    def test_frozen_payload_authenticates_group_assignments_and_folds(self):
        provenance = {"development_sha256": "a" * 64}
        topics = helpful_topics()

        first = freeze_policy(topics, provenance=provenance)
        regrouped = list(topics)
        regrouped[0] = DevelopmentRecord(
            group="treebench:images/regrouped.jpg",
            backbone=topics[0].backbone,
            benchmark=topics[0].benchmark,
            ordinal=topics[0].ordinal,
            stage2_row=topics[0].stage2_row,
            split_row=topics[0].split_row,
            calibration=topics[0].calibration,
        )
        second = freeze_policy(tuple(regrouped), provenance=provenance)

        self.assertNotEqual(
            first["source_group_assignments_sha256"],
            second["source_group_assignments_sha256"],
        )
        self.assertNotEqual(first["oof_folds_sha256"], second["oof_folds_sha256"])


if __name__ == "__main__":
    unittest.main()
