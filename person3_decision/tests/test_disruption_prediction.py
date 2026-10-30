"""Tests for the explicit operational-prediction availability contract."""

import unittest

from person3_decision.disruption_prediction import (
    DISRUPTION_PREDICTION_UNAVAILABLE,
    DisruptionPredictionStatus,
)


class DisruptionPredictionAvailabilityTests(unittest.TestCase):
    def test_prediction_is_unavailable_without_operational_outcomes(self):
        prediction = DISRUPTION_PREDICTION_UNAVAILABLE

        self.assertEqual(
            prediction.status,
            DisruptionPredictionStatus.UNAVAILABLE,
        )
        self.assertIsNone(prediction.probability)
        self.assertIn("no verified", prediction.reason)
        self.assertTrue(
            any("prediction horizon" in item for item in prediction.required_data)
        )
        self.assertTrue(
            any("outcome labels" in item for item in prediction.required_data)
        )


if __name__ == "__main__":
    unittest.main()
