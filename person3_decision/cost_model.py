"""Cost calculations for supplier-allocation scenarios.

Only purchase costs and explicitly supplied supplier-switching fees are
included. Other cost categories are outside the scope of this first version.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

Money = Decimal
Quantity = Decimal


@dataclass(frozen=True)
class CostEstimate:
    """Known cost totals and explicit markers for unavailable cost data."""

    purchase_cost_by_supplier: Mapping[str, Money | None]
    purchase_cost: Money | None
    switching_cost_by_supplier: Mapping[str, Money | None]
    switching_cost: Money | None
    total_cost: Money | None
    complete: bool
    unknown_components: tuple[str, ...]


def calculate_cost(
    allocations: Mapping[str, Quantity],
    unit_costs: Mapping[str, Money | None],
    switching_costs: Mapping[str, Money | None],
    current_supplier_id: str,
) -> CostEstimate:
    """Calculate purchase and switching costs for an allocation.

    A missing unit cost makes the purchase total unknown. A missing switching
    fee makes the switching total unknown only when that non-current supplier
    is used. An explicitly supplied Decimal("0") is a known zero, not missing.

    Switching fees are assumed to apply once per used non-current supplier.
    The caller must supply fees on that basis; no transport, holding,
    stockout, or disruption-loss costs are included.
    """

    purchase_by_supplier: dict[str, Money | None] = {}
    switching_by_supplier: dict[str, Money | None] = {}
    unknown: list[str] = []

    for supplier_id, quantity in allocations.items():
        unit_cost = unit_costs.get(supplier_id)
        if unit_cost is None:
            purchase_by_supplier[supplier_id] = None
            unknown.append(f"purchase_cost:{supplier_id}")
        else:
            purchase_by_supplier[supplier_id] = quantity * unit_cost

        if supplier_id == current_supplier_id:
            switching_by_supplier[supplier_id] = Decimal("0")
            continue

        fee = switching_costs.get(supplier_id)
        if fee is None:
            switching_by_supplier[supplier_id] = None
            unknown.append(f"switching_cost:{supplier_id}")
        else:
            switching_by_supplier[supplier_id] = fee

    purchase_total = (
        None
        if any(value is None for value in purchase_by_supplier.values())
        else sum(purchase_by_supplier.values(), Decimal("0"))
    )
    switching_total = (
        None
        if any(value is None for value in switching_by_supplier.values())
        else sum(switching_by_supplier.values(), Decimal("0"))
    )
    total = (
        purchase_total + switching_total
        if purchase_total is not None and switching_total is not None
        else None
    )

    return CostEstimate(
        purchase_cost_by_supplier=purchase_by_supplier,
        purchase_cost=purchase_total,
        switching_cost_by_supplier=switching_by_supplier,
        switching_cost=switching_total,
        total_cost=total,
        complete=total is not None,
        unknown_components=tuple(unknown),
    )
