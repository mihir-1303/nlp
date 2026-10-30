"""Evaluate supply scenarios using explicitly supplied order data.

Cost-only ranking is the default. Risk-aware selection is unavailable until
supplier/scenario risk inputs and an approved conversion policy exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Mapping, Sequence

from .cost_model import CostEstimate, calculate_cost
from .disruption_prediction import (
    DISRUPTION_PREDICTION_UNAVAILABLE,
    DisruptionPrediction,
)
from .risk_scoring import RiskAssessment

Number = Decimal | int | float | str


def _decimal(value: Number, field_name: str) -> Decimal:
    """Convert a numeric input to a finite Decimal with a readable error."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number, not a boolean.")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite.")
    return result


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _non_negative(value: Number, field_name: str) -> Decimal:
    result = _decimal(value, field_name)
    if result < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return result


@dataclass(frozen=True)
class SupplierOption:
    """Operational facts supplied for one supplier option.

    ``unit_cost`` and ``switching_cost`` may be ``None`` when unknown.
    A supplied numeric zero is treated as a confirmed zero.
    """

    supplier_id: str
    available_quantity: Number
    lead_time_days: Number
    unit_cost: Number | None
    switching_cost: Number | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "supplier_id", _required_text(self.supplier_id, "supplier_id")
        )
        object.__setattr__(
            self,
            "available_quantity",
            _non_negative(self.available_quantity, "available_quantity"),
        )
        object.__setattr__(
            self,
            "lead_time_days",
            _non_negative(self.lead_time_days, "lead_time_days"),
        )
        if self.unit_cost is not None:
            object.__setattr__(
                self, "unit_cost", _non_negative(self.unit_cost, "unit_cost")
            )
        if self.switching_cost is not None:
            object.__setattr__(
                self,
                "switching_cost",
                _non_negative(self.switching_cost, "switching_cost"),
            )


@dataclass(frozen=True)
class OrderInput:
    """Order identity and required quantity information."""

    product_id: str
    order_quantity: Number
    quantity_unit: str
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "product_id", _required_text(self.product_id, "product_id")
        )
        quantity = _decimal(self.order_quantity, "order_quantity")
        if quantity <= 0:
            raise ValueError("order_quantity must be greater than zero.")
        object.__setattr__(self, "order_quantity", quantity)
        object.__setattr__(
            self, "quantity_unit", _required_text(self.quantity_unit, "quantity_unit")
        )
        object.__setattr__(
            self, "currency", _required_text(self.currency, "currency").upper()
        )


@dataclass(frozen=True)
class Constraints:
    """Optional decision constraints; omitted limits are not assumed."""

    max_lead_time_days: Number | None = None
    max_budget: Number | None = None
    approved_supplier_ids: frozenset[str] | None = None
    max_suppliers_per_order: int | None = None

    def __post_init__(self) -> None:
        if self.max_lead_time_days is not None:
            object.__setattr__(
                self,
                "max_lead_time_days",
                _non_negative(self.max_lead_time_days, "max_lead_time_days"),
            )
        if self.max_budget is not None:
            object.__setattr__(
                self, "max_budget", _non_negative(self.max_budget, "max_budget")
            )
        if self.approved_supplier_ids is not None:
            approved = frozenset(
                _required_text(item, "approved_supplier_ids[]")
                for item in self.approved_supplier_ids
            )
            object.__setattr__(self, "approved_supplier_ids", approved)
        if self.max_suppliers_per_order is not None:
            if (
                isinstance(self.max_suppliers_per_order, bool)
                or not isinstance(self.max_suppliers_per_order, int)
                or self.max_suppliers_per_order < 1
            ):
                raise ValueError("max_suppliers_per_order must be a positive integer.")


class IncidentAssociationStatus(str, Enum):
    """Whether an incident is explicitly associated with this evaluation."""

    UNLINKED = "unlinked"
    USER_CONFIRMED = "user_confirmed_for_evaluation"


@dataclass(frozen=True)
class IncidentRiskContext:
    """Optional incident context, with user confirmation distinct from data linkage.

    ``risk_assessment`` may be ``None`` or unassessed; neither means low risk.
    ``USER_CONFIRMED`` records a user's association to this evaluation only. It
    is not a verified incident/order relationship from project data.
    """

    association_status: IncidentAssociationStatus
    risk_assessment: RiskAssessment | None

    def __post_init__(self) -> None:
        if not isinstance(self.association_status, IncidentAssociationStatus):
            raise ValueError(
                "association_status must be an IncidentAssociationStatus."
            )
        if self.risk_assessment is not None and not isinstance(
            self.risk_assessment, RiskAssessment
        ):
            raise ValueError("risk_assessment must be a RiskAssessment or None.")


@dataclass(frozen=True)
class EvaluationInput:
    """Complete input for evaluating one order and its supplier options."""

    order: OrderInput
    current_supplier: SupplierOption
    alternative_suppliers: tuple[SupplierOption, ...] = ()
    constraints: Constraints = field(default_factory=Constraints)
    incident_context: IncidentRiskContext | None = None

    def __post_init__(self) -> None:
        alternatives = tuple(self.alternative_suppliers)
        object.__setattr__(self, "alternative_suppliers", alternatives)
        supplier_ids = [self.current_supplier.supplier_id] + [
            supplier.supplier_id for supplier in alternatives
        ]
        if len(supplier_ids) != len(set(supplier_ids)):
            raise ValueError("Supplier IDs must be unique within an evaluation.")
        if self.incident_context is not None and not isinstance(
            self.incident_context, IncidentRiskContext
        ):
            raise ValueError("incident_context must be an IncidentRiskContext or None.")


class FeasibilityStatus(str, Enum):
    """Whether a scenario satisfies constraints that can be evaluated."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


class DecisionPolicy(str, Enum):
    """Requested scenario-selection policy."""

    COST_ONLY = "cost_only"
    RISK_AWARE = "risk_aware"


class RiskComparisonStatus(str, Enum):
    """Whether candidate scenario risk can be compared under a policy."""

    NOT_REQUESTED = "not_requested"
    NOT_COMPARABLE = "not_comparable"


class RecommendationStatus(str, Enum):
    """Why a decision result does or does not contain a selected scenario."""

    UNSPECIFIED = "unspecified"
    COST_ONLY_RECOMMENDATION = "cost_only_recommendation"
    NO_FEASIBLE_SCENARIO = "no_feasible_scenario"
    NO_COMPLETE_COST_SCENARIO = "no_complete_cost_scenario"
    NO_ELIGIBLE_SCENARIO = "no_eligible_scenario"
    RISK_COMPARISON_UNAVAILABLE = "risk_comparison_unavailable"


@dataclass(frozen=True)
class Allocation:
    """Quantity assigned to a supplier in a scenario."""

    supplier_id: str
    quantity: Decimal


@dataclass(frozen=True)
class ScenarioResult:
    """One generated scenario and its feasibility/cost evaluation."""

    scenario_id: str
    action: str
    allocations: tuple[Allocation, ...]
    feasibility: FeasibilityStatus
    infeasibility_reasons: tuple[str, ...]
    feasibility_unknown_reasons: tuple[str, ...]
    cost: CostEstimate


@dataclass(frozen=True)
class Recommendation:
    """Selected scenario plus a transparent cost-only rationale."""

    scenario_id: str | None
    action: str | None
    allocations: tuple[Allocation, ...]
    cost: CostEstimate | None
    selection_rule: str
    explanation: str


@dataclass(frozen=True)
class DecisionResult:
    """All scenarios and recommendation metadata for one evaluation."""

    product_id: str
    order_quantity: Decimal
    quantity_unit: str
    currency: str
    scenarios: tuple[ScenarioResult, ...]
    recommendation: Recommendation
    recommendation_policy: str = "cost_only"
    residual_risk: str = "not_assessed"
    assumptions: tuple[str, ...] = (
        "Switching fees apply once per used non-current supplier.",
        "Only purchase and explicitly supplied switching costs are included.",
    )
    limitations: tuple[str, ...] = (
        "Cost-only is the default; risk-aware comparison is unavailable.",
        "Transportation, inventory holding, stockout, and disruption-loss "
        "costs are not included.",
        "Person 1 event-classifier confidence is not used as disruption "
        "probability.",
        "Person 2 predictions are not required or consumed.",
        "Feasibility covers only the supplied constraints and supplier data.",
        "Incident context is informational and does not affect scenario ranking.",
    )
    incident_context: IncidentRiskContext | None = None
    risk_comparison_status: RiskComparisonStatus = (
        RiskComparisonStatus.NOT_REQUESTED
    )
    risk_comparison_reasons: tuple[str, ...] = ()
    recommendation_status: RecommendationStatus = RecommendationStatus.UNSPECIFIED
    disruption_prediction: DisruptionPrediction = (
        DISRUPTION_PREDICTION_UNAVAILABLE
    )

    @property
    def has_feasible_scenario(self) -> bool:
        """Whether any scenario is explicitly feasible under supplied data."""

        return any(
            scenario.feasibility is FeasibilityStatus.FEASIBLE
            for scenario in self.scenarios
        )


def _scenario_cost(
    allocations: tuple[Allocation, ...],
    suppliers: Mapping[str, SupplierOption],
    current_supplier_id: str,
) -> CostEstimate:
    quantities = {item.supplier_id: item.quantity for item in allocations}
    return calculate_cost(
        allocations=quantities,
        unit_costs={
            supplier_id: supplier.unit_cost
            for supplier_id, supplier in suppliers.items()
        },
        switching_costs={
            supplier_id: supplier.switching_cost
            for supplier_id, supplier in suppliers.items()
        },
        current_supplier_id=current_supplier_id,
    )


def _evaluate_scenario(
    scenario_id: str,
    action: str,
    allocations: tuple[Allocation, ...],
    evaluation: EvaluationInput,
    suppliers: Mapping[str, SupplierOption],
) -> ScenarioResult:
    reasons: list[str] = []
    unknown_reasons: list[str] = []
    constraints = evaluation.constraints
    used_ids = [item.supplier_id for item in allocations]

    for allocation in allocations:
        supplier = suppliers[allocation.supplier_id]
        if allocation.quantity > supplier.available_quantity:
            reasons.append(
                f"Supplier {supplier.supplier_id} capacity "
                f"{supplier.available_quantity} is below allocated quantity "
                f"{allocation.quantity}."
            )
        if (
            constraints.approved_supplier_ids is not None
            and supplier.supplier_id not in constraints.approved_supplier_ids
        ):
            reasons.append(
                f"Supplier {supplier.supplier_id} is not in the approved supplier list."
            )

    if (
        constraints.max_suppliers_per_order is not None
        and len(used_ids) > constraints.max_suppliers_per_order
    ):
        reasons.append(
            f"Scenario uses {len(used_ids)} suppliers, exceeding the maximum "
            f"of {constraints.max_suppliers_per_order}."
        )

    if constraints.max_lead_time_days is not None:
        violating = [
            suppliers[supplier_id]
            for supplier_id in used_ids
            if suppliers[supplier_id].lead_time_days
            > constraints.max_lead_time_days
        ]
        for supplier in violating:
            reasons.append(
                f"Supplier {supplier.supplier_id} lead time "
                f"{supplier.lead_time_days} days exceeds the maximum "
                f"{constraints.max_lead_time_days} days."
            )

    cost = _scenario_cost(allocations, suppliers, evaluation.current_supplier.supplier_id)
    if constraints.max_budget is not None:
        if cost.total_cost is None:
            unknown_reasons.append(
                "Cannot verify the maximum budget because scenario cost is incomplete."
            )
        elif cost.total_cost > constraints.max_budget:
            reasons.append(
                f"Scenario cost {cost.total_cost} exceeds maximum budget "
                f"{constraints.max_budget}."
            )

    if reasons:
        feasibility = FeasibilityStatus.INFEASIBLE
    elif unknown_reasons:
        feasibility = FeasibilityStatus.UNKNOWN
    else:
        feasibility = FeasibilityStatus.FEASIBLE

    return ScenarioResult(
        scenario_id=scenario_id,
        action=action,
        allocations=allocations,
        feasibility=feasibility,
        infeasibility_reasons=tuple(reasons),
        feasibility_unknown_reasons=tuple(unknown_reasons),
        cost=cost,
    )


def _validate_explicit_allocations(
    explicit_allocations: Sequence[Mapping[str, Number]],
    evaluation: EvaluationInput,
    suppliers: Mapping[str, SupplierOption],
) -> tuple[tuple[Allocation, ...], ...]:
    validated: list[tuple[Allocation, ...]] = []
    required_quantity = evaluation.order.order_quantity

    for index, allocation_map in enumerate(explicit_allocations, start=1):
        if not allocation_map:
            raise ValueError(f"Split allocation {index} must not be empty.")
        allocation_items: list[Allocation] = []
        for supplier_id, raw_quantity in allocation_map.items():
            supplier_id = _required_text(supplier_id, "allocation supplier_id")
            if supplier_id not in suppliers:
                raise ValueError(
                    f"Split allocation {index} references unknown supplier "
                    f"{supplier_id!r}."
                )
            quantity = _decimal(raw_quantity, f"allocation[{supplier_id}]")
            if quantity <= 0:
                raise ValueError(
                    f"Allocated quantity for {supplier_id} must be greater than zero."
                )
            allocation_items.append(Allocation(supplier_id, quantity))
        total = sum((item.quantity for item in allocation_items), Decimal("0"))
        if total != required_quantity:
            raise ValueError(
                f"Split allocation {index} sums to {total}; it must equal "
                f"the order quantity {required_quantity}."
            )
        validated.append(tuple(sorted(allocation_items, key=lambda item: item.supplier_id)))
    return tuple(validated)


def evaluate_order(
    evaluation: EvaluationInput,
    split_allocations: Sequence[Mapping[str, Number]] = (),
    policy: DecisionPolicy | str = DecisionPolicy.COST_ONLY,
) -> DecisionResult:
    """Evaluate scenarios using cost-only ranking or report risk as incomparable.

    Split allocations are mappings from supplier ID to allocated quantity.
    Malformed split allocations raise ``ValueError``. Capacity and constraint
    violations instead remain visible as scenario results. The risk-aware
    policy is deliberately blocked until supplier/scenario risk inputs and an
    approved risk-to-cost conversion contract exist; it never falls back to
    cost-only ranking or treats unknown risk as zero.
    """

    try:
        selected_policy = DecisionPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "policy must be 'cost_only' or 'risk_aware'."
        ) from exc

    suppliers = {
        evaluation.current_supplier.supplier_id: evaluation.current_supplier,
        **{
            supplier.supplier_id: supplier
            for supplier in evaluation.alternative_suppliers
        },
    }
    full_quantity = evaluation.order.order_quantity
    candidates: list[tuple[str, str, tuple[Allocation, ...]]] = [
        (
            "continue_current_supplier",
            "continue_with_current_supplier",
            (Allocation(evaluation.current_supplier.supplier_id, full_quantity),),
        )
    ]

    for supplier in evaluation.alternative_suppliers:
        candidates.append(
            (
                f"switch_to_{supplier.supplier_id}",
                "switch_supplier",
                (Allocation(supplier.supplier_id, full_quantity),),
            )
        )

    validated_splits = _validate_explicit_allocations(
        split_allocations, evaluation, suppliers
    )
    for index, allocations in enumerate(validated_splits, start=1):
        candidates.append((f"explicit_split_{index:03d}", "split_order", allocations))

    scenarios = tuple(
        _evaluate_scenario(
            scenario_id=scenario_id,
            action=action,
            allocations=allocations,
            evaluation=evaluation,
            suppliers=suppliers,
        )
        for scenario_id, action, allocations in candidates
    )

    eligible = [
        scenario
        for scenario in scenarios
        if scenario.feasibility is FeasibilityStatus.FEASIBLE
        and scenario.cost.complete
        and scenario.cost.total_cost is not None
    ]
    selection_rule = (
        "Recommend the lowest total cost among feasible scenarios with complete "
        "cost estimates; break exact cost ties by lexicographically smallest "
        "scenario_id."
    )
    risk_comparison_status = RiskComparisonStatus.NOT_REQUESTED
    risk_comparison_reasons: tuple[str, ...] = ()
    recommendation_status: RecommendationStatus
    if selected_policy is DecisionPolicy.RISK_AWARE:
        selection_rule = (
            "Risk-aware comparison is unavailable. No risk-adjusted winner "
            "is selected until the required risk inputs and approved conversion "
            "policy are implemented."
        )
        risk_comparison_status = RiskComparisonStatus.NOT_COMPARABLE
        reasons = [
            "Supplier- and scenario-specific risk inputs are not available.",
            "An approved risk-to-cost conversion policy has not been defined.",
        ]
        context = evaluation.incident_context
        if (
            context is None
            or context.association_status is IncidentAssociationStatus.UNLINKED
        ):
            reasons.append(
                "No user-confirmed incident association is available for this evaluation."
            )
        if (
            context is None
            or context.risk_assessment is None
            or context.risk_assessment.score is None
        ):
            reasons.append(
                "A complete incident risk assessment is not available; "
                "unassessed risk is not treated as zero."
            )
        risk_comparison_reasons = tuple(reasons)
        recommendation_status = RecommendationStatus.RISK_COMPARISON_UNAVAILABLE
        recommendation = Recommendation(
            scenario_id=None,
            action=None,
            allocations=(),
            cost=None,
            selection_rule=selection_rule,
            explanation=(
                "No risk-aware recommendation was made because scenario risk "
                "is not comparable with the available inputs. The cost and "
                "feasibility results remain visible, but are not presented as "
                "a risk-aware ranking."
            ),
        )
    elif eligible:
        selected = min(
            eligible,
            key=lambda scenario: (
                scenario.cost.total_cost,
                scenario.scenario_id,
            ),
        )
        recommendation = Recommendation(
            scenario_id=selected.scenario_id,
            action=selected.action,
            allocations=selected.allocations,
            cost=selected.cost,
            selection_rule=selection_rule,
            explanation=(
                f"Selected {selected.scenario_id}: it has the lowest complete "
                "estimated cost among scenarios satisfying the supplied "
                "constraints. This is a cost-only recommendation; residual "
                "disruption risk was not assessed."
            ),
        )
        recommendation_status = RecommendationStatus.COST_ONLY_RECOMMENDATION
    else:
        if scenarios and all(
            scenario.feasibility is FeasibilityStatus.INFEASIBLE
            for scenario in scenarios
        ):
            recommendation_status = RecommendationStatus.NO_FEASIBLE_SCENARIO
        elif any(
            scenario.feasibility is FeasibilityStatus.FEASIBLE
            for scenario in scenarios
        ):
            recommendation_status = (
                RecommendationStatus.NO_COMPLETE_COST_SCENARIO
            )
        else:
            recommendation_status = RecommendationStatus.NO_ELIGIBLE_SCENARIO
        recommendation = Recommendation(
            scenario_id=None,
            action=None,
            allocations=(),
            cost=None,
            selection_rule=selection_rule,
            explanation=(
                "No recommendation was made because no scenario is both "
                "feasible under the supplied constraints and supported by a "
                "complete cost estimate."
            ),
        )

    return DecisionResult(
        product_id=evaluation.order.product_id,
        order_quantity=evaluation.order.order_quantity,
        quantity_unit=evaluation.order.quantity_unit,
        currency=evaluation.order.currency,
        scenarios=scenarios,
        recommendation=recommendation,
        incident_context=evaluation.incident_context,
        recommendation_policy=selected_policy.value,
        risk_comparison_status=risk_comparison_status,
        risk_comparison_reasons=risk_comparison_reasons,
        recommendation_status=recommendation_status,
        disruption_prediction=DISRUPTION_PREDICTION_UNAVAILABLE,
    )
