"""Tests for the provisional heuristic incident risk index."""

import unittest

from person3_decision.risk_scoring import (
    RiskLevel,
    RiskScoringConfig,
    score_incident,
)


class RiskScoringTests(unittest.TestCase):
    def test_scores_known_category_and_severity(self):
        result = score_incident("Weather", "Severe")

        self.assertEqual(result.score, 70)
        self.assertEqual(result.level, RiskLevel.HIGH)
        self.assertEqual(result.category_points, 25)
        self.assertEqual(result.severity_points, 45)
        self.assertIn("not a calibrated probability", result.notes[0])

    def test_normalizes_case_and_surrounding_whitespace(self):
        result = score_incident(" weather ", " moderate ")

        self.assertEqual(result.score, 50)
        self.assertEqual(result.level, RiskLevel.HIGH)

    def test_missing_category_or_severity_is_unassessed(self):
        for category, severity in (
            (None, "Severe"),
            ("Weather", None),
            (" ", "Minor"),
            ("Accident", ""),
        ):
            with self.subTest(category=category, severity=severity):
                result = score_incident(category, severity)
                self.assertIsNone(result.score)
                self.assertEqual(result.level, RiskLevel.UNASSESSED)
                self.assertTrue(result.notes)

    def test_unknown_category_or_severity_is_unassessed(self):
        for category, severity in (
            ("Unlisted Event", "Severe"),
            ("Weather", "Unlisted Severity"),
        ):
            with self.subTest(category=category, severity=severity):
                result = score_incident(category, severity)
                self.assertIsNone(result.score)
                self.assertEqual(result.level, RiskLevel.UNASSESSED)

    def test_zero_points_is_a_valid_low_score(self):
        config = RiskScoringConfig(
            category_points={"Example": 0},
            severity_points={"Minor": 0},
        )

        result = score_incident("Example", "Minor", config)

        self.assertEqual(result.score, 0)
        self.assertEqual(result.level, RiskLevel.LOW)

    def test_each_risk_band_boundary_is_inclusive(self):
        config = RiskScoringConfig(
            category_points={"Example": 0},
            severity_points={
                "Low edge": 24,
                "Medium edge": 25,
                "Medium upper": 49,
                "High edge": 50,
                "High upper": 74,
                "Critical edge": 75,
            },
        )
        expected = (
            ("Low edge", RiskLevel.LOW),
            ("Medium edge", RiskLevel.MEDIUM),
            ("Medium upper", RiskLevel.MEDIUM),
            ("High edge", RiskLevel.HIGH),
            ("High upper", RiskLevel.HIGH),
            ("Critical edge", RiskLevel.CRITICAL),
        )

        for severity, level in expected:
            with self.subTest(severity=severity):
                result = score_incident("Example", severity, config)
                self.assertEqual(result.score, config.severity_points[severity])
                self.assertEqual(result.level, level)

    def test_score_is_capped_at_configured_maximum(self):
        config = RiskScoringConfig(
            category_points={"Example": 90},
            severity_points={"Extreme": 90},
        )

        result = score_incident("Example", "Extreme", config)

        self.assertEqual(result.score, 100)
        self.assertEqual(result.level, RiskLevel.CRITICAL)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Risk band limits"):
            RiskScoringConfig(low_max=50, medium_max=40)
        with self.assertRaisesRegex(ValueError, "integer from 0"):
            RiskScoringConfig(category_points={"Example": -1})


if __name__ == "__main__":
    unittest.main()
