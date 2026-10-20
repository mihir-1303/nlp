"""Dashboard tests use synthetic values only; they are not operational data."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from person3_decision.app import _build_split_allocations


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
        app.text_input[0].set_value("SYNTHETIC-PRODUCT")
        app.text_input[1].set_value("10")
        app.text_input[2].set_value("units")
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

        app.button[0].click()
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

        app.button[0].click()
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

        app.button[0].click()
        app.run(timeout=15)

        self.assertFalse(app.exception)
        self.assertTrue(
            any("must equal the order quantity" in item.value for item in app.error)
        )
        self.assertNotIn("explicit_split_", self._rendered_text(app))


if __name__ == "__main__":
    unittest.main()
