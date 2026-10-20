"""Run a self-contained synthetic demonstration of the decision engine.

This demo uses only synthetic values defined here. It does not access project
datasets, saved models, or external services.
"""

from __future__ import annotations

from decimal import Decimal

from person3_decision.decision_engine import (
    Constraints,
    EvaluationInput,
    OrderInput,
    SupplierOption,
    evaluate_order,
)


def _format_money(value: Decimal | None, currency: str) -> str:
    """Format a known money amount or label it as unknown."""

    if value is None:
        return "UNKNOWN"
    return f"{currency} {value:.2f}"


def main() -> None:
    """Evaluate and print synthetic sourcing scenarios."""

    print("SYNTHETIC DEMONSTRATION - values below are not project data.")
    print("Recommendation policy: COST-ONLY. Residual risk: NOT ASSESSED.\n")

    evaluation = EvaluationInput(
        order=OrderInput(
            product_id="SYNTHETIC-PRODUCT-001",
            order_quantity=100,
            quantity_unit="units",
            currency="USD",
        ),
        current_supplier=SupplierOption(
            supplier_id="SYNTHETIC-CURRENT",
            available_quantity=100,
            lead_time_days=8,
            unit_cost=Decimal("12.00"),
            switching_cost=None,
        ),
        alternative_suppliers=(
            SupplierOption(
                supplier_id="SYNTHETIC-ALTERNATIVE-A",
                available_quantity=100,
                lead_time_days=6,
                unit_cost=Decimal("11.00"),
                switching_cost=Decimal("0"),
            ),
            SupplierOption(
                supplier_id="SYNTHETIC-ALTERNATIVE-B",
                available_quantity=75,
                lead_time_days=5,
                unit_cost=Decimal("10.50"),
                switching_cost=Decimal("50.00"),
            ),
        ),
        constraints=Constraints(
            max_lead_time_days=7,
            max_budget=Decimal("1200.00"),
            max_suppliers_per_order=2,
        ),
    )
    explicit_split = {
        "SYNTHETIC-ALTERNATIVE-A": 60,
        "SYNTHETIC-ALTERNATIVE-B": 40,
    }
    result = evaluate_order(
        evaluation,
        split_allocations=(explicit_split,),
    )

    print(
        f"Order: {result.product_id}; quantity: "
        f"{result.order_quantity} {result.quantity_unit}; currency: {result.currency}"
    )
    print("\nSCENARIOS")
    for scenario in result.scenarios:
        print(f"\n- {scenario.scenario_id} ({scenario.action})")
        print(f"  Feasibility: {scenario.feasibility.value}")
        if scenario.infeasibility_reasons:
            print(
                "  Infeasibility reasons: "
                + "; ".join(scenario.infeasibility_reasons)
            )
        if scenario.feasibility_unknown_reasons:
            print(
                "  Feasibility unknown: "
                + "; ".join(scenario.feasibility_unknown_reasons)
            )
        allocation_summary = ", ".join(
            f"{allocation.supplier_id}={allocation.quantity} "
            f"{result.quantity_unit}"
            for allocation in scenario.allocations
        )
        print(f"  Allocations: {allocation_summary}")

        cost = scenario.cost
        print(f"  Cost complete: {'yes' if cost.complete else 'no'}")
        print(f"  Purchase cost: {_format_money(cost.purchase_cost, result.currency)}")
        for supplier_id, amount in cost.purchase_cost_by_supplier.items():
            print(
                f"    Purchase from {supplier_id}: "
                f"{_format_money(amount, result.currency)}"
            )
        print(f"  Switching cost: {_format_money(cost.switching_cost, result.currency)}")
        for supplier_id, amount in cost.switching_cost_by_supplier.items():
            print(
                f"    Switching fee for {supplier_id}: "
                f"{_format_money(amount, result.currency)}"
            )
        print(f"  Total estimated cost: {_format_money(cost.total_cost, result.currency)}")
        if cost.unknown_components:
            print("  Unknown cost components: " + ", ".join(cost.unknown_components))
        print(
            "  Included cost categories: purchase and explicitly supplied "
            "applicable switching fees only."
        )

    recommendation = result.recommendation
    print("\nRECOMMENDATION")
    if recommendation.scenario_id is None:
        print("No recommendation available.")
    else:
        print(f"Scenario: {recommendation.scenario_id}")
        print(f"Action: {recommendation.action}")
        print(
            "Allocations: "
            + ", ".join(
                f"{allocation.supplier_id}={allocation.quantity} "
                f"{result.quantity_unit}"
                for allocation in recommendation.allocations
            )
        )
        if recommendation.cost is not None:
            print(
                "Estimated total cost: "
                f"{_format_money(recommendation.cost.total_cost, result.currency)}"
            )
    print(f"Selection rule: {recommendation.selection_rule}")
    print(f"Explanation: {recommendation.explanation}")
    print(f"Policy: {result.recommendation_policy}; risk: {result.residual_risk}.")
    print("This example does not assess disruption probability or residual risk.")


if __name__ == "__main__":
    main()
