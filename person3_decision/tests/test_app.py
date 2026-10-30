"""Dashboard tests use synthetic values only; they are not operational data."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from person3_decision.app import _build_split_allocations
from person3_decision.decision_engine import (
    DecisionPolicy,
    IncidentAssociationStatus,
    RecommendationStatus,
    evaluate_order,
)
from person3_decision.late_delivery_prediction import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from person3_decision.risk_scoring import score_incident


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class SplitAllocationModeTests(unittest.TestCase):
    def test_disabled_mode_does_not_parse_or_return_split_values(self):
        allocations = _build_split_allocations(
            enabled=False,
            supplier_ids=("CURRENT", "SUP-A"),
            split_values=("not-a-number",),
        )
        self.assertEqual(allocations, ())

    def test_enabled_mode_parses_explicit_allocations(self):
        allocations = _build_split_allocations(
            enabled=True,
            supplier_ids=("CURRENT", "SUP-A"),
            split_values=("2", "8"),
        )
        self.assertEqual(
            allocations,
            ({"CURRENT": 2, "SUP-A": 8},),
        )

    def test_enabled_mode_rejects_invalid_split_quantity(self):
        with self.assertRaisesRegex(ValueError, "SUP-A"):
            _build_split_allocations(
                enabled=True,
                supplier_ids=("CURRENT", "SUP-A"),
                split_values=("2", "invalid"),
            )


class DashboardSplitModeTests(unittest.TestCase):
    def _app_with_order_inputs(self) -> AppTest:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
        app.text_input(key="decision_product_id").set_value("SYNTHETIC-PRODUCT")
        app.text_input(key="decision_order_quantity").set_value("10")
        app.text_input(key="decision_quantity_unit").set_value("units")
        supplier_values = {
            "current_id": "CURRENT",
            "current_capacity": "10",
            "current_lead_time": "4",
            "current_unit_cost": "20",
            "alternative_0_id": "SUP-A",
            "alternative_0_capacity": "10",
            "alternative_0_lead_time": "3",
            "alternative_0_unit_cost": "50",
            "alternative_1_id": "SUP-B",
            "alternative_1_capacity": "10",
            "alternative_1_lead_time": "3",
            "alternative_1_unit_cost": "12",
        }
        for key, value in supplier_values.items():
            app.text_input(key=key).set_value(value)
        app.checkbox(key="alternative_0_switching_cost_unknown").uncheck()
        app.text_input(key="alternative_0_switching_cost").set_value("0")
        app.checkbox(key="alternative_1_switching_cost_unknown").uncheck()
        app.text_input(key="alternative_1_switching_cost").set_value("0")
        return app

    def _rendered_text(self, app: AppTest) -> str:
        values = [
            item.value
            for collection_name in ("success", "subheader", "warning", "error")
            for item in getattr(app, collection_name)
        ]
        return "\n".join(str(value) for value in values)

    def test_split_disabled_by_default_and_not_evaluated(self):
        app = self._app_with_order_inputs()
        split_toggle = app.checkbox(key="enable_split_evaluation")
        self.assertFalse(split_toggle.value)
        self.assertFalse(
            any(item.label.startswith("Split quantity") for item in app.text_input)
        )

        app.button(key="evaluate_scenarios_button").click()
        app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertIn("switch_to_SUP-B", self._rendered_text(app))
        self.assertNotIn("explicit_split_", self._rendered_text(app))

    def test_split_enabled_and_valid_allocation_is_evaluated(self):
        app = self._app_with_order_inputs()
        app.checkbox(key="enable_split_evaluation").check()
        app.run(timeout=15)
        app.text_input(key="split_0").set_value("")
        app.text_input(key="split_1").set_value("5")
        app.text_input(key="split_2").set_value("5")

        app.button(key="evaluate_scenarios_button").click()
        app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertIn("explicit_split_001", self._rendered_text(app))
        self.assertFalse(app.error)

    def test_split_enabled_requires_allocations_sum_to_order_quantity(self):
        app = self._app_with_order_inputs()
        app.checkbox(key="enable_split_evaluation").check()
        app.run(timeout=15)
        app.text_input(key="split_0").set_value("")
        app.text_input(key="split_1").set_value("2")
        app.text_input(key="split_2").set_value("2")

        app.button(key="evaluate_scenarios_button").click()
        app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("must equal the order quantity" in item.value for item in app.error)
        )
        self.assertNotIn("explicit_split_", self._rendered_text(app))

    def test_user_confirmed_incident_context_reaches_evaluation(self):
        app = self._app_with_order_inputs()
        app.session_state["incident_risk_assessment"] = score_incident(
            "Weather",
            None,
        )
        app.run(timeout=15)

        association_checkbox = app.checkbox(
            key="confirm_incident_order_association"
        )
        self.assertFalse(association_checkbox.value)
        association_checkbox.check()
        app.run(timeout=15)

        with patch(
            "person3_decision.decision_engine.evaluate_order",
            wraps=evaluate_order,
        ) as evaluate:
            app.button(key="evaluate_scenarios_button").click()
            app.run(timeout=15)

        self.assertFalse(app.exception)
        evaluation = evaluate.call_args.args[0]
        self.assertEqual(
            evaluation.incident_context.association_status,
            IncidentAssociationStatus.USER_CONFIRMED,
        )
        self.assertIsNone(evaluation.incident_context.risk_assessment.score)

    def test_risk_aware_selector_is_opt_in_and_withholds_winner(self):
        app = self._app_with_order_inputs()
        policy_selector = app.selectbox(key="decision_policy")
        self.assertEqual(policy_selector.value, DecisionPolicy.COST_ONLY)
        policy_selector.select(DecisionPolicy.RISK_AWARE)
        app.run(timeout=30)

        with patch(
            "person3_decision.decision_engine.evaluate_order",
            wraps=evaluate_order,
        ) as evaluate:
            app.button(key="evaluate_scenarios_button").click()
            app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertEqual(
            evaluate.call_args.kwargs["policy"],
            DecisionPolicy.RISK_AWARE,
        )
        self.assertTrue(
            any(
                "Risk-aware comparison unavailable" in item.value
                for item in app.warning
            )
        )
        self.assertFalse(
            any("Recommended scenario" in item.value for item in app.success)
        )

    def test_unspecified_status_never_confirms_populated_or_missing_scenario(self):
        for scenario_id in (None, "switch_to_SUP-B"):
            with self.subTest(scenario_id=scenario_id):
                app = self._app_with_order_inputs()
                app.run(timeout=30)

                def unspecified_result(*args, **kwargs):
                    result = evaluate_order(*args, **kwargs)
                    recommendation = replace(
                        result.recommendation,
                        scenario_id=scenario_id,
                    )
                    return replace(
                        result,
                        recommendation=recommendation,
                        recommendation_status=RecommendationStatus.UNSPECIFIED,
                    )

                with patch(
                    "person3_decision.decision_engine.evaluate_order",
                    side_effect=unspecified_result,
                ):
                    app.button(key="evaluate_scenarios_button").click()
                    app.run(timeout=30)

                self.assertFalse(app.exception)
                self.assertTrue(
                    any(
                        "Recommendation status is unspecified/unconfirmed"
                        in item.value
                        for item in app.warning
                    )
                )
                self.assertFalse(
                    any(
                        "RECOMMENDED:" in item.value
                        or "Recommended scenario:" in item.value
                        for item in app.success
                    )
                )

    def test_no_complete_cost_scenario_has_distinct_ui_message(self):
        app = self._app_with_order_inputs()
        for key in (
            "current_unit_cost_unknown",
            "alternative_0_unit_cost_unknown",
            "alternative_1_unit_cost_unknown",
        ):
            app.checkbox(key=key).check()

        app.run(timeout=30)
        app.button(key="evaluate_scenarios_button").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Feasible scenarios exist, but none has a complete cost estimate"
                in item.value
                for item in app.warning
            )
        )

    def test_operational_prediction_unavailable_is_separate_from_incident_index(self):
        app = self._app_with_order_inputs()
        app.button(key="evaluate_scenarios_button").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Operational disruption prediction: Unavailable."
                in item.value
                for item in app.warning
            )
        )
        self.assertTrue(
            any(
                "event-category classification and the heuristic incident risk index"
                in item.value
                for item in app.caption
            )
        )
        self.assertTrue(
            any(
                "no verified order-, shipment-, or supplier-linked"
                in item.value
                for item in app.markdown
            )
        )

    def test_no_feasible_scenario_is_reported_distinctly(self):
        app = self._app_with_order_inputs()
        app.checkbox(key="max_budget_enabled").check()
        app.text_input(key="max_budget").set_value("100")
        app.run(timeout=30)

        app.button(key="evaluate_scenarios_button").click()
        app.run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "No feasible supplier scenario" in item.value
                for item in app.warning
            )
        )
        self.assertFalse(
            any("Risk-aware comparison unavailable" in item.value for item in app.warning)
        )

    def test_risk_aware_unavailable_is_distinct_from_no_feasible_scenario(self):
        app = self._app_with_order_inputs()
        app.checkbox(key="max_budget_enabled").check()
        app.text_input(key="max_budget").set_value("100")
        app.selectbox(key="decision_policy").select(DecisionPolicy.RISK_AWARE)
        app.run(timeout=30)

        app.button(key="evaluate_scenarios_button").click()
        app.run(timeout=30)

        warnings = [item.value for item in app.warning]
        self.assertFalse(app.exception)
        self.assertTrue(
            any("Risk-aware comparison unavailable" in warning for warning in warnings)
        )
        self.assertTrue(
            any("No supplier scenario is confirmed feasible" in warning for warning in warnings)
        )


class DashboardLateDeliveryPredictionTests(unittest.TestCase):
    def _prediction_app(self) -> AppTest:
        return AppTest.from_file(str(APP_PATH)).run(timeout=15)

    def _fill_prediction_inputs(self, app: AppTest) -> None:
        values = (
            "2",
            "2",
            "100",
            "10",
            "0.10",
            "12.9716",
            "77.5946",
            "DEBIT",
            "Standard Class",
            "Pacific Asia",
            "Southeast Asia",
            "Cleats",
            "Footwear",
            "Consumer",
        )
        self.assertEqual(len(values), len(NUMERIC_FEATURES + CATEGORICAL_FEATURES))
        for index, value in enumerate(values):
            app.text_input(key=f"late_delivery_input_{index}").set_value(value)

    def test_prediction_is_displayed_separately_from_supplier_decisions(self):
        class StubModel:
            classes_ = (0, 1)

            def __init__(self):
                self.received_features = None

            def predict_proba(self, features):
                self.received_features = features
                return [[0.2, 0.8]]

        app = self._prediction_app()
        self._fill_prediction_inputs(app)
        app.run(timeout=15)
        model = StubModel()

        with (
            patch(
                "person3_decision.late_delivery_prediction.load_late_delivery_model",
                return_value=model,
            ),
            patch(
                "person3_decision.app.evaluate_order",
                wraps=evaluate_order,
            ) as evaluate,
        ):
            app.button(key="late_delivery_prediction_button").click()
            app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        self.assertEqual(
            app.metric[0].label,
            "Predicted Late_delivery_risk probability",
        )
        self.assertEqual(app.metric[0].value, "80.0%")
        self.assertTrue(
            any("selected 0.40 prototype threshold" in item.value for item in app.warning)
        )
        self.assertTrue(
            any("not a calibrated probability cutoff" in item.value for item in app.caption)
        )
        self.assertEqual(
            list(model.received_features.columns),
            list(NUMERIC_FEATURES + CATEGORICAL_FEATURES),
        )
        evaluate.assert_not_called()

    def test_warning_threshold_is_configurable(self):
        class StubModel:
            classes_ = (0, 1)

            def predict_proba(self, features):
                return [[0.2, 0.8]]

        app = self._prediction_app()
        self._fill_prediction_inputs(app)
        app.number_input(key="late_delivery_warning_threshold").set_value(0.9)
        app.run(timeout=15)

        with patch(
            "person3_decision.late_delivery_prediction.load_late_delivery_model",
            return_value=StubModel(),
        ):
            app.button(key="late_delivery_prediction_button").click()
            app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("below the selected 0.90 prototype threshold" in item.value for item in app.info)
        )

    def test_prediction_failure_is_shown_without_crashing_dashboard(self):
        class BrokenModel:
            classes_ = (0, 1)

            def predict_proba(self, features):
                raise OSError("test prediction failure")

        app = self._prediction_app()
        self._fill_prediction_inputs(app)
        app.run(timeout=15)

        with patch(
            "person3_decision.late_delivery_prediction.load_late_delivery_model",
            return_value=BrokenModel(),
        ):
            app.button(key="late_delivery_prediction_button").click()
            app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Late-delivery prediction unavailable" in item.value
                and "test prediction failure" in item.value
                for item in app.error
            )
        )

    def test_missing_order_inputs_are_shown_as_form_errors(self):
        app = self._prediction_app()
        app.button(key="late_delivery_prediction_button").click()
        app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Late-delivery prediction unavailable" in item.value
                and "missing or blank" in item.value
                for item in app.error
            )
        )

    def test_model_loading_error_is_shown_without_crashing_dashboard(self):
        app = self._prediction_app()
        self._fill_prediction_inputs(app)
        app.run(timeout=15)

        with patch(
            "person3_decision.late_delivery_prediction.load_late_delivery_model",
            side_effect=RuntimeError("test model-load failure"),
        ):
            app.button(key="late_delivery_prediction_button").click()
            app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Late-delivery prediction unavailable" in item.value
                and "test model-load failure" in item.value
                for item in app.error
            )
        )


if __name__ == "__main__":
    unittest.main()
