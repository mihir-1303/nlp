"""Tests use synthetic values only; they are not project operational data."""

from decimal import Decimal
import unittest

from person3_decision.cost_model import calculate_cost
from person3_decision.decision_engine import (
    Constraints,
    EvaluationInput,
    FeasibilityStatus,
    OrderInput,
    SupplierOption,
    evaluate_order,
)


def synthetic_evaluation(
    *,
    current_cost=10,
    alternative_cost=8,
    current_capacity=100,
    alternative_capacity=100,
    current_lead_time=5,
    alternative_lead_time=4,
    current_switching_cost=None,
    alternative_switching_cost=3,
    constraints=None,
):
    """Build a small, explicitly synthetic supplier/order fixture."""

    return EvaluationInput(
        order=OrderInput(
            product_id="SYNTHETIC-PRODUCT",
            order_quantity=10,
            quantity_unit="units",
            currency="USD",
        ),
        current_supplier=SupplierOption(
            supplier_id="SYNTHETIC-CURRENT",
            available_quantity=current_capacity,
            lead_time_days=current_lead_time,
            unit_cost=current_cost,
            switching_cost=current_switching_cost,
        ),
        alternative_suppliers=(
            SupplierOption(
                supplier_id="SYNTHETIC-ALT",
                available_quantity=alternative_capacity,
                lead_time_days=alternative_lead_time,
                unit_cost=alternative_cost,
                switching_cost=alternative_switching_cost,
            ),
        ),
        constraints=constraints or Constraints(),
    )


class CostModelTests(unittest.TestCase):
    def test_purchase_cost_for_single_supplier(self):
        estimate = calculate_cost(
            allocations={"A": Decimal("4")},
            unit_costs={"A": Decimal("2.50")},
            switching_costs={"A": None},
            current_supplier_id="A",
        )
        self.assertEqual(estimate.purchase_cost, Decimal("10.00"))
        self.assertEqual(estimate.switching_cost, Decimal("0"))
        self.assertEqual(estimate.total_cost, Decimal("10.00"))
        self.assertTrue(estimate.complete)

    def test_purchase_cost_for_split_order(self):
        estimate = calculate_cost(
            allocations={"A": Decimal("4"), "B": Decimal("6")},
            unit_costs={"A": Decimal("2"), "B": Decimal("3")},
            switching_costs={"A": None, "B": Decimal("0")},
            current_supplier_id="A",
        )
        self.assertEqual(estimate.purchase_cost, Decimal("26"))
        self.assertEqual(estimate.switching_cost, Decimal("0"))
        self.assertEqual(estimate.total_cost, Decimal("26"))
        self.assertTrue(estimate.complete)

    def test_zero_switching_fee_is_known_but_none_is_unknown(self):
        known_zero = calculate_cost(
            allocations={"ALT": Decimal("1")},
            unit_costs={"ALT": Decimal("5")},
            switching_costs={"ALT": Decimal("0")},
            current_supplier_id="CURRENT",
        )
        unknown = calculate_cost(
            allocations={"ALT": Decimal("1")},
            unit_costs={"ALT": Decimal("5")},
            switching_costs={"ALT": None},
            current_supplier_id="CURRENT",
        )
        self.assertEqual(known_zero.switching_cost, Decimal("0"))
        self.assertTrue(known_zero.complete)
        self.assertIsNone(unknown.switching_cost)
        self.assertIsNone(unknown.total_cost)
        self.assertFalse(unknown.complete)

    def test_missing_cost_mapping_entry_is_unknown(self):
        estimate = calculate_cost(
            allocations={"ALT": Decimal("1")},
            unit_costs={},
            switching_costs={},
            current_supplier_id="CURRENT",
        )
        self.assertFalse(estimate.complete)
        self.assertIsNone(estimate.purchase_cost)
        self.assertIsNone(estimate.switching_cost)


class DecisionEngineTests(unittest.TestCase):
    def test_invalid_order_quantity_is_rejected(self):
        for quantity in (0, -1):
            with self.subTest(quantity=quantity), self.assertRaisesRegex(
                ValueError, "order_quantity"
            ):
                OrderInput("P", quantity, "units", "USD")

    def test_invalid_supplier_cost_capacity_and_lead_time_are_rejected(self):
        invalid_fields = (
            {"unit_cost": -1},
            {"available_quantity": -1},
            {"lead_time_days": -1},
            {"switching_cost": -1},
        )
        for override in invalid_fields:
            values = {
                "supplier_id": "S",
                "available_quantity": 5,
                "lead_time_days": 2,
                "unit_cost": 1,
                "switching_cost": 0,
            }
            values.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                SupplierOption(**values)

    def test_duplicate_supplier_ids_are_rejected(self):
        evaluation = synthetic_evaluation()
        duplicate = SupplierOption(
            supplier_id=evaluation.current_supplier.supplier_id,
            available_quantity=10,
            lead_time_days=1,
            unit_cost=1,
            switching_cost=0,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            EvaluationInput(
                order=evaluation.order,
                current_supplier=evaluation.current_supplier,
                alternative_suppliers=(duplicate,),
            )

    def test_unknown_split_supplier_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown supplier"):
            evaluate_order(
                synthetic_evaluation(),
                split_allocations=({"UNKNOWN": 10},),
            )

    def test_split_must_sum_to_order_quantity(self):
        with self.assertRaisesRegex(ValueError, "must equal"):
            evaluate_order(
                synthetic_evaluation(),
                split_allocations=({"SYNTHETIC-CURRENT": 4},),
            )

    def test_negative_split_allocation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            evaluate_order(
                synthetic_evaluation(),
                split_allocations=(
                    {"SYNTHETIC-CURRENT": -1, "SYNTHETIC-ALT": 11},
                ),
            )

    def test_capacity_violation_is_retained_as_infeasible(self):
        result = evaluate_order(
            synthetic_evaluation(alternative_capacity=5)
        )
        alternative = next(
            scenario for scenario in result.scenarios
            if scenario.scenario_id == "switch_to_SYNTHETIC-ALT"
        )
        self.assertEqual(alternative.feasibility, FeasibilityStatus.INFEASIBLE)
        self.assertIn("capacity", alternative.infeasibility_reasons[0])

    def test_supplied_constraints_are_enforced(self):
        result = evaluate_order(
            synthetic_evaluation(
                constraints=Constraints(
                    max_lead_time_days=3,
                    max_budget=50,
                    approved_supplier_ids=frozenset({"SYNTHETIC-CURRENT"}),
                    max_suppliers_per_order=1,
                )
            ),
            split_allocations=(
                {"SYNTHETIC-CURRENT": 5, "SYNTHETIC-ALT": 5},
            ),
        )
        current, alternative, split = result.scenarios
        self.assertEqual(current.feasibility, FeasibilityStatus.INFEASIBLE)
        self.assertTrue(any("lead time" in reason for reason in current.infeasibility_reasons))
        self.assertEqual(alternative.feasibility, FeasibilityStatus.INFEASIBLE)
        self.assertTrue(any("approved" in reason for reason in alternative.infeasibility_reasons))
        self.assertEqual(split.feasibility, FeasibilityStatus.INFEASIBLE)
        self.assertTrue(any("maximum of 1" in reason for reason in split.infeasibility_reasons))

    def test_lowest_cost_feasible_scenario_is_selected(self):
        result = evaluate_order(synthetic_evaluation())
        self.assertEqual(
            result.recommendation.scenario_id,
            "switch_to_SYNTHETIC-ALT",
        )
        self.assertEqual(
            result.recommendation.cost.total_cost,
            Decimal("83"),
        )

    def test_exact_cost_tie_uses_lexicographically_smallest_scenario_id(self):
        evaluation = synthetic_evaluation(
            current_cost=20,
            alternative_cost=10,
            alternative_switching_cost=0,
        )
        second_alternative = SupplierOption(
            supplier_id="SYNTHETIC-ZETA",
            available_quantity=100,
            lead_time_days=4,
            unit_cost=10,
            switching_cost=0,
        )
        evaluation = EvaluationInput(
            order=evaluation.order,
            current_supplier=evaluation.current_supplier,
            alternative_suppliers=(
                second_alternative,
                evaluation.alternative_suppliers[0],
            ),
        )
        result = evaluate_order(evaluation)
        self.assertEqual(
            result.recommendation.scenario_id,
            "switch_to_SYNTHETIC-ALT",
        )

    def test_cost_incomplete_scenarios_are_excluded(self):
        evaluation = synthetic_evaluation(alternative_switching_cost=None)
        result = evaluate_order(evaluation)
        alternative = next(
            scenario for scenario in result.scenarios
            if scenario.scenario_id == "switch_to_SYNTHETIC-ALT"
        )
        self.assertFalse(alternative.cost.complete)
        self.assertEqual(
            result.recommendation.scenario_id,
            "continue_current_supplier",
        )

    def test_unknown_budget_feasibility_is_not_assumed(self):
        result = evaluate_order(
            synthetic_evaluation(
                alternative_cost=None,
                constraints=Constraints(max_budget=1000),
            )
        )
        alternative = next(
            scenario for scenario in result.scenarios
            if scenario.scenario_id == "switch_to_SYNTHETIC-ALT"
        )
        self.assertEqual(alternative.feasibility, FeasibilityStatus.UNKNOWN)
        self.assertTrue(alternative.feasibility_unknown_reasons)

    def test_maximum_budget_violation_is_infeasible(self):
        result = evaluate_order(
            synthetic_evaluation(
                constraints=Constraints(max_budget=50)
            )
        )
        current = result.scenarios[0]
        alternative = result.scenarios[1]
        self.assertEqual(current.feasibility, FeasibilityStatus.INFEASIBLE)
        self.assertTrue(any("maximum budget" in reason for reason in current.infeasibility_reasons))
        self.assertEqual(alternative.feasibility, FeasibilityStatus.INFEASIBLE)

    def test_no_recommendation_when_no_eligible_scenario_exists(self):
        result = evaluate_order(
            synthetic_evaluation(
                current_cost=None,
                alternative_cost=None,
                constraints=Constraints(max_budget=100),
            )
        )
        self.assertIsNone(result.recommendation.scenario_id)
        self.assertIsNone(result.recommendation.cost)

    def test_residual_risk_is_not_assessed(self):
        result = evaluate_order(synthetic_evaluation())
        self.assertEqual(result.residual_risk, "not_assessed")
        self.assertEqual(result.recommendation_policy, "cost_only")

    def test_nlp_confidence_is_not_an_engine_input_or_risk_probability(self):
        result = evaluate_order(synthetic_evaluation())
        self.assertNotIn(
            "event_prediction_confidence",
            result.recommendation.explanation,
        )
        self.assertTrue(
            any("not used as disruption" in item for item in result.limitations)
        )


if __name__ == "__main__":
    unittest.main()
