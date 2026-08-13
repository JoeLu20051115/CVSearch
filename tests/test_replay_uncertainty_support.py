import math
import unittest

from cvsearch.eval.replay_uncertainty_support import (
    PROFILES,
    AdvantageFeatures,
    UtilityIsotonicCalibrator,
    fit_utility_isotonic,
    raw_advantage,
)


class UtilityPrimitiveTests(unittest.TestCase):
    def test_pava_accepts_continuous_targets_and_is_monotone(self):
        fitted = fit_utility_isotonic(
            ((0.1, 0.75), (0.2, 0.25), (0.3, 1.0)),
        )

        predictions = [fitted.predict(value) for value in (0.1, 0.2, 0.3)]

        self.assertEqual(predictions, sorted(predictions))
        self.assertEqual(predictions, [0.5, 0.5, 1.0])

    def test_equal_scores_are_averaged_before_pava(self):
        fitted = fit_utility_isotonic(
            ((0.1, 0.25), (0.1, 0.75), (0.2, 1.0)),
        )

        self.assertEqual(fitted.upper_bounds, (0.1, 0.2))
        self.assertEqual(fitted.utilities, (0.5, 1.0))

    def test_raw_advantage_is_the_only_weighted_numeric_score(self):
        features = AdvantageFeatures(0.8, 0.5, 0.7, 0.6, 0.4)

        self.assertAlmostEqual(
            raw_advantage(features, PROFILES["balanced"]),
            0.6,
        )

    def test_invalid_feature_or_profile_is_rejected(self):
        for bad in (-0.1, 1.1, math.inf, math.nan, True):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    AdvantageFeatures(bad, 0.5, 0.5, 0.5, 0.5)

        with self.assertRaises(ValueError):
            raw_advantage(
                AdvantageFeatures(0.5, 0.5, 0.5, 0.5, 0.5),
                (1.0, 0.0, 0.0, 0.0, 0.1),
            )

    def test_calibrator_rejects_bad_samples_and_payload(self):
        for samples in (
            (),
            ((math.nan, 0.5),),
            ((0.1, -0.1),),
            ((0.1, 1.1),),
            ((True, 0.5),),
        ):
            with self.subTest(samples=samples):
                with self.assertRaises((TypeError, ValueError)):
                    fit_utility_isotonic(samples)

        for calibrator in (
            UtilityIsotonicCalibrator((), ()),
            UtilityIsotonicCalibrator((0.2, 0.1), (0.4, 0.5)),
            UtilityIsotonicCalibrator((0.1,), (1.1,)),
        ):
            with self.subTest(calibrator=calibrator):
                with self.assertRaises(ValueError):
                    calibrator.predict(0.1)

    def test_calibrator_round_trip_is_deterministic(self):
        fitted = fit_utility_isotonic(((0.2, 0.7), (0.1, 0.3)))

        self.assertEqual(
            UtilityIsotonicCalibrator.from_dict(fitted.to_dict()),
            fitted,
        )
        self.assertEqual(fitted.to_dict(), fitted.to_dict())


if __name__ == "__main__":
    unittest.main()
