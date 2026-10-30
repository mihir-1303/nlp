"""Demo output tests use synthetic decision results only."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import unittest
from unittest.mock import patch

from person3_decision.decision_engine import (
    RecommendationStatus,
    evaluate_order,
)
from person3_decision.demo import main


class DemoRecommendationStatusTests(unittest.TestCase):
    def test_unspecified_status_is_explicitly_unconfirmed(self):
        for scenario_id in (None, "synthetic-scenario"):
            with self.subTest(scenario_id=scenario_id):
                def unspecified_result(*args, **kwargs):
                    result = evaluate_order(*args, **kwargs)
                    return replace(
                        result,
                        recommendation=replace(
                            result.recommendation,
                            scenario_id=scenario_id,
                        ),
                        recommendation_status=RecommendationStatus.UNSPECIFIED,
                    )

                output = StringIO()
                with (
                    patch(
                        "person3_decision.demo.evaluate_order",
                        side_effect=unspecified_result,
                    ),
                    redirect_stdout(output),
                ):
                    main()

                self.assertIn(
                    "Recommendation status unspecified/unconfirmed",
                    output.getvalue(),
                )
                self.assertNotIn("No eligible scenario", output.getvalue())
                self.assertNotIn("Scenario: synthetic-scenario", output.getvalue())


if __name__ == "__main__":
    unittest.main()
