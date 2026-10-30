"""Tests for the separate order-time late-delivery prediction capability."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from person3_decision.late_delivery_prediction import (
    FEATURES,
    MODEL_PATH,
    LateDeliveryPrediction,
    load_late_delivery_model,
    predict_late_delivery,
)


SAMPLE_ORDER = {
    "Days for shipment (scheduled)": 2,
    "Order Item Quantity": 2,
    "Order Item Product Price": 100.0,
    "Order Item Discount": 10.0,
    "Order Item Discount Rate": 0.10,
    "Latitude": 12.9716,
    "Longitude": 77.5946,
    "Type": "DEBIT",
    "Shipping Mode": "Standard Class",
    "Market": "Pacific Asia",
    "Order Region": "Southeast Asia",
    "Category Name": "Cleats",
    "Department Name": "Footwear",
    "Customer Segment": "Consumer",
}


class StubModel:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities=(0.6, 0.4)):
        self.probabilities = probabilities
        self.received_features = None

    def predict_proba(self, features):
        self.received_features = features
        return np.array([self.probabilities])


class LateDeliveryPredictionTests(unittest.TestCase):
    def test_artifact_path_is_independent_of_working_directory(self):
        self.assertEqual(
            MODEL_PATH,
            Path(__file__).resolve().parents[2]
            / "person1_nlp"
            / "supply-chain-prediction"
            / "late_delivery_model.joblib",
        )

    def test_successful_prediction_uses_raw_features_and_selected_threshold(self):
        model = StubModel(probabilities=(0.2, 0.8))

        result = predict_late_delivery(
            SAMPLE_ORDER,
            warning_threshold=0.4,
            model=model,
        )

        self.assertEqual(
            result,
            LateDeliveryPrediction(
                probability=0.8,
                warning_threshold=0.4,
                warning=True,
            ),
        )
        self.assertEqual(list(model.received_features.columns), list(FEATURES))
        self.assertEqual(
            model.received_features.iloc[0]["Shipping Mode"],
            "Standard Class",
        )

    def test_saved_pipeline_predicts_successfully(self):
        prediction = predict_late_delivery(SAMPLE_ORDER)

        self.assertGreaterEqual(prediction.probability, 0)
        self.assertLessEqual(prediction.probability, 1)
        self.assertEqual(prediction.warning_threshold, 0.4)
        self.assertEqual(prediction.warning, prediction.probability >= 0.4)

    def test_missing_and_blank_required_values_are_rejected(self):
        for update in (
            {"Type": None},
            {"Type": "  "},
            {"Latitude": None},
        ):
            order = {**SAMPLE_ORDER, **update}
            with self.subTest(update=update), self.assertRaisesRegex(
                ValueError,
                "missing or blank",
            ):
                predict_late_delivery(order, model=StubModel())

    def test_invalid_numeric_values_and_thresholds_are_rejected(self):
        for feature, value in (
            ("Days for shipment (scheduled)", "invalid"),
            ("Order Item Quantity", 0),
            ("Order Item Discount Rate", 1.1),
            ("Latitude", 91),
            ("Longitude", float("inf")),
        ):
            order = {**SAMPLE_ORDER, feature: value}
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                predict_late_delivery(order, model=StubModel())

        for threshold in (-0.1, 1.1, float("nan")):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(
                ValueError,
                "Warning threshold",
            ):
                predict_late_delivery(
                    SAMPLE_ORDER,
                    warning_threshold=threshold,
                    model=StubModel(),
                )

    def test_model_loading_failure_is_reported(self):
        with patch(
            "person3_decision.late_delivery_prediction.joblib.load",
            side_effect=OSError("corrupt artifact"),
        ), self.assertRaisesRegex(RuntimeError, "Could not load late-delivery model"):
            load_late_delivery_model(MODEL_PATH)

    def test_missing_model_file_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "Late-delivery model not found"):
            load_late_delivery_model(MODEL_PATH.with_name("missing-model.joblib"))

    def test_prediction_failure_is_reported(self):
        class BrokenModel(StubModel):
            def predict_proba(self, features):
                raise OSError("inference failed")

        with self.assertRaisesRegex(RuntimeError, "Late-delivery prediction failed"):
            predict_late_delivery(SAMPLE_ORDER, model=BrokenModel())


if __name__ == "__main__":
    unittest.main()
