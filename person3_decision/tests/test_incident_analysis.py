"""Tests for the separate incident event-classification dashboard feature."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import joblib
from streamlit.testing.v1 import AppTest

from person3_decision.app import classify_incident, load_event_classifier


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class IncidentClassificationTests(unittest.TestCase):
    def test_classifies_text_as_an_event_category(self):
        model = Mock()
        model.predict.return_value = ["Weather"]

        category = classify_incident(
            "  Heavy rainfall caused flooding and delayed transportation.  ",
            model=model,
        )

        self.assertEqual(category, "Weather")
        model.predict.assert_called_once_with(
            ["Heavy rainfall caused flooding and delayed transportation."]
        )

    def test_rejects_blank_text_without_loading_model(self):
        with patch(
            "person3_decision.app.load_event_classifier"
        ) as load_model:
            with self.assertRaisesRegex(ValueError, "Enter an incident description"):
                classify_incident(" \n ")

        load_model.assert_not_called()

    def test_reports_missing_model(self):
        with patch(
            "person3_decision.app.load_event_classifier",
            side_effect=FileNotFoundError("model file is missing"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "model file is missing"):
                classify_incident("A port closure delayed shipments.")

    def test_reports_inference_failure(self):
        model = Mock()
        model.predict.side_effect = ValueError("invalid input")

        with self.assertRaisesRegex(RuntimeError, "Incident classification failed"):
            classify_incident("A port closure delayed shipments.", model=model)

    def test_dashboard_displays_blank_text_validation_error(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.button(key="classify_incident_button").click()
        app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Enter an incident description" in item.value
                for item in app.error
            )
        )

    def test_dashboard_displays_model_loading_error(self):
        load_event_classifier.clear()
        try:
            with patch.object(
                joblib,
                "load",
                side_effect=OSError("synthetic missing artifact"),
            ):
                app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
                app.text_area(key="incident_description").set_value(
                    "A port closure delayed shipments."
                )
                app.button(key="classify_incident_button").click()
                app.run(timeout=15)

            self.assertFalse(app.exception)
            self.assertTrue(
                any(
                    "Could not load event classifier model" in item.value
                    and "synthetic missing artifact" in item.value
                    for item in app.error
                )
            )
        finally:
            load_event_classifier.clear()


if __name__ == "__main__":
    unittest.main()
