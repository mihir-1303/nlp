"""Streamlit dashboard for incident classification and sourcing scenarios.

Operational values are entered by the user, and supplier recommendations
remain cost-only with residual disruption risk unassessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol, Sequence

import joblib
import streamlit as st

from person3_decision.decision_engine import (
    Constraints,
    DecisionPolicy,
    DecisionResult,
    EvaluationInput,
    IncidentAssociationStatus,
    IncidentRiskContext,
    OrderInput,
    RecommendationStatus,
    ScenarioResult,
    SupplierOption,
    evaluate_order,
)
from person3_decision.late_delivery_prediction import (
    CATEGORICAL_FEATURES,
    DEFAULT_WARNING_THRESHOLD,
    NUMERIC_FEATURES,
    predict_late_delivery,
)
from person3_decision.risk_scoring import RiskAssessment, score_incident

EVENT_CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[1]
    / "person1_nlp"
    / "models"
    / "supply_chain_event_pipeline.joblib"
)


class EventClassifier(Protocol):
    """Prediction interface used by the saved event-classification pipeline."""

    def predict(self, texts: list[str]) -> Sequence[str]: ...


@st.cache_resource
def load_event_classifier() -> EventClassifier:
    """Load the saved Person 1 event classifier once per Streamlit process."""

    if not EVENT_CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(
            f"Event classifier model not found: {EVENT_CLASSIFIER_PATH}"
        )

    try:
        return joblib.load(EVENT_CLASSIFIER_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load event classifier model at "
            f"{EVENT_CLASSIFIER_PATH}: {exc}"
        ) from exc


def classify_incident(
    description: str,
    model: EventClassifier | None = None,
) -> str:
    """Return one event category without deriving a risk score or probability."""

    if not description.strip():
        raise ValueError("Enter an incident description before classifying it.")

    classifier = model if model is not None else load_event_classifier()
    try:
        predictions = classifier.predict([description.strip()])
        if len(predictions) != 1:
            raise ValueError("The classifier did not return exactly one category.")
        category = str(predictions[0]).strip()
        if not category:
            raise ValueError("The classifier returned an empty event category.")
        return category
    except Exception as exc:
        raise RuntimeError(f"Incident classification failed: {exc}") from exc


def _build_incident_risk_context(
    assessment: RiskAssessment | None,
    association_confirmed: bool,
) -> IncidentRiskContext | None:
    """Represent user-confirmed or unlinked incident context without ranking it."""

    if assessment is None:
        return None
    association_status = (
        IncidentAssociationStatus.USER_CONFIRMED
        if association_confirmed
        else IncidentAssociationStatus.UNLINKED
    )
    return IncidentRiskContext(
        association_status=association_status,
        risk_assessment=assessment,
    )


@dataclass(frozen=True)
class SupplierFormData:
    """Raw supplier widget values, validated only on form submission."""

    label: str
    supplier_id: str
    available_quantity: str
    lead_time_days: str
    unit_cost_unknown: bool
    unit_cost_text: str
    switching_cost_unknown: bool
    switching_cost_text: str


def _parse_decimal(value: str, field_name: str) -> Decimal:
    """Parse a required decimal string and report field-specific errors."""

    try:
        number = Decimal(value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc
    if not number.is_finite():
        raise ValueError(f"{field_name} must be finite.")
    return number


def _required_text(value: str, field_name: str) -> str:
    """Reject blank text values before creating the engine input."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required.")
    return normalized


def _money(value: Decimal | None, currency: str) -> str:
    """Format a cost amount or explicitly label it as unknown."""

    if value is None:
        return "Unknown"
    return f"{currency} {value:,.2f}"


def _supplier_fields(
    label: str,
    key: str,
    *,
    switching_cost_applicable: bool,
) -> SupplierFormData:
    """Render supplier fields and retain values for submit-time validation."""

    st.markdown(f"**{label}**")
    supplier_id = st.text_input(
        "Supplier ID",
        key=f"{key}_id",
        placeholder="Enter a supplier identifier",
    )
    available_quantity = st.text_input(
        "Available quantity",
        key=f"{key}_capacity",
        placeholder="Enter quantity in the order unit",
    )
    lead_time_days = st.text_input(
        "Lead time (days)",
        key=f"{key}_lead_time",
        placeholder="Enter lead time",
    )

    unit_cost_unknown = st.checkbox(
        "Unit cost unknown",
        key=f"{key}_unit_cost_unknown",
        value=False,
    )
    unit_cost_text = st.text_input(
        "Unit cost (per quantity unit)",
        key=f"{key}_unit_cost",
        placeholder="Enter a supplied cost, including 0 for confirmed zero",
        disabled=unit_cost_unknown,
    )

    if switching_cost_applicable:
        switching_cost_unknown = st.checkbox(
            "Switching cost unknown",
            key=f"{key}_switching_cost_unknown",
            value=True,
        )
        switching_cost_text = st.text_input(
            "One-time switching cost",
            key=f"{key}_switching_cost",
            placeholder="Enter a supplied fee, including 0 for confirmed zero",
            disabled=switching_cost_unknown,
        )
    else:
        st.caption("No switching fee is applied to the current supplier.")
        switching_cost_unknown = True
        switching_cost_text = ""

    return SupplierFormData(
        label=label,
        supplier_id=supplier_id,
        available_quantity=available_quantity,
        lead_time_days=lead_time_days,
        unit_cost_unknown=unit_cost_unknown,
        unit_cost_text=unit_cost_text,
        switching_cost_unknown=switching_cost_unknown,
        switching_cost_text=switching_cost_text,
    )


def _build_supplier(
    form_data: SupplierFormData,
    *,
    switching_cost_applicable: bool,
) -> SupplierOption:
    """Validate raw widget values and create an engine supplier option."""

    return SupplierOption(
        supplier_id=_required_text(
            form_data.supplier_id,
            f"{form_data.label} supplier ID",
        ),
        available_quantity=_parse_decimal(
            form_data.available_quantity,
            f"{form_data.label} available quantity",
        ),
        lead_time_days=_parse_decimal(
            form_data.lead_time_days,
            f"{form_data.label} lead time",
        ),
        unit_cost=(
            None
            if form_data.unit_cost_unknown
            else _parse_decimal(form_data.unit_cost_text, f"{form_data.label} unit cost")
        ),
        switching_cost=(
            None
            if not switching_cost_applicable or form_data.switching_cost_unknown
            else _parse_decimal(
                form_data.switching_cost_text,
                f"{form_data.label} switching cost",
            )
        ),
    )


def _render_scenario(
    scenario: ScenarioResult,
    currency: str,
    quantity_unit: str,
    recommended_scenario_id: str | None,
) -> None:
    """Render scenario feasibility, allocations, and itemized cost results."""

    is_recommended = scenario.scenario_id == recommended_scenario_id
    heading = f"{scenario.scenario_id} — {scenario.action}"
    with st.container(border=True):
        if is_recommended:
            st.success(f"RECOMMENDED: {heading}")
        else:
            st.subheader(heading)

        st.write(f"**Feasibility:** `{scenario.feasibility.value}`")
        if scenario.infeasibility_reasons:
            st.error("Infeasible: " + "; ".join(scenario.infeasibility_reasons))
        if scenario.feasibility_unknown_reasons:
            st.warning(
                "Feasibility could not be fully verified: "
                + "; ".join(scenario.feasibility_unknown_reasons)
            )

        st.write("**Supplier allocations**")
        st.dataframe(
            [
                {
                    "Supplier ID": allocation.supplier_id,
                    f"Quantity ({quantity_unit})": str(allocation.quantity),
                }
                for allocation in scenario.allocations
            ],
            hide_index=True,
            width="stretch",
        )

        cost = scenario.cost
        st.write(f"**Cost complete:** {'Yes' if cost.complete else 'No'}")
        st.write(f"**Purchase cost:** {_money(cost.purchase_cost, currency)}")
        st.dataframe(
            [
                {
                    "Supplier ID": supplier_id,
                    "Purchase cost": _money(amount, currency),
                }
                for supplier_id, amount in cost.purchase_cost_by_supplier.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.write(f"**Switching cost:** {_money(cost.switching_cost, currency)}")
        st.dataframe(
            [
                {
                    "Supplier ID": supplier_id,
                    "Switching cost": _money(amount, currency),
                }
                for supplier_id, amount in cost.switching_cost_by_supplier.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.write(f"**Total estimated cost:** {_money(cost.total_cost, currency)}")
        if cost.unknown_components:
            st.caption("Unknown cost components: " + ", ".join(cost.unknown_components))
        st.caption(
            "Cost includes purchase and applicable switching fees only; "
            "transport, inventory holding, stockout, and disruption-loss costs "
            "are not included."
        )


def _build_evaluation(
    product_id: str,
    order_quantity: str,
    quantity_unit: str,
    currency: str,
    current_supplier_data: SupplierFormData,
    alternative_supplier_data: Sequence[SupplierFormData],
    *,
    max_lead_time_enabled: bool,
    max_lead_time: str,
    max_budget_enabled: bool,
    max_budget: str,
    approved_suppliers_enabled: bool,
    approved_supplier_ids_text: str,
    max_suppliers_enabled: bool,
    max_suppliers: int,
    incident_context: IncidentRiskContext | None = None,
) -> EvaluationInput:
    """Build engine input using only entered values and enabled constraints."""

    approved_supplier_ids: frozenset[str] | None = None
    if approved_suppliers_enabled:
        identifiers = {
            item.strip()
            for item in approved_supplier_ids_text.split(",")
            if item.strip()
        }
        if not identifiers:
            raise ValueError(
                "Enter at least one approved supplier ID or disable that constraint."
            )
        approved_supplier_ids = frozenset(identifiers)

    current_supplier = _build_supplier(
        current_supplier_data,
        switching_cost_applicable=False,
    )
    alternative_suppliers = tuple(
        _build_supplier(data, switching_cost_applicable=True)
        for data in alternative_supplier_data
    )
    constraints = Constraints(
        max_lead_time_days=(
            _parse_decimal(max_lead_time, "Maximum lead time")
            if max_lead_time_enabled
            else None
        ),
        max_budget=(
            _parse_decimal(max_budget, "Maximum budget")
            if max_budget_enabled
            else None
        ),
        approved_supplier_ids=approved_supplier_ids,
        max_suppliers_per_order=max_suppliers if max_suppliers_enabled else None,
    )

    return EvaluationInput(
        order=OrderInput(
            product_id=_required_text(product_id, "Product ID"),
            order_quantity=_parse_decimal(order_quantity, "Order quantity"),
            quantity_unit=_required_text(quantity_unit, "Quantity unit"),
            currency=_required_text(currency, "Currency"),
        ),
        current_supplier=current_supplier,
        alternative_suppliers=tuple(alternative_suppliers),
        constraints=constraints,
        incident_context=incident_context,
    )


def _build_split_allocations(
    enabled: bool,
    supplier_ids: Sequence[str],
    split_values: Sequence[str],
) -> tuple[dict[str, Decimal], ...]:
    """Build explicit split inputs only when split evaluation is enabled."""

    if not enabled:
        return ()

    allocation = {
        supplier_id: _parse_decimal(
            value,
            f"Split quantity for {supplier_id}",
        )
        for supplier_id, value in zip(supplier_ids, split_values)
        if value.strip()
    }
    return (allocation,)


def _render_late_delivery_prediction() -> None:
    """Render a standalone late-delivery prediction form and result."""

    st.subheader("Late-delivery prediction (order-time inputs)")
    st.caption(
        "This saved model estimates the dataset target Late_delivery_risk for "
        "the supplied order attributes. It is separate from incident "
        "classification, the heuristic incident-risk index, supplier-risk "
        "estimates, and the supplier decision policy."
    )
    st.caption(
        "The 0.4 default is a selected prototype operating threshold, not a "
        "calibrated probability cutoff or a proven business optimum. This "
        "late-delivery estimate is not a general supply-chain disruption "
        "probability and is not linked to an incident or supplier."
    )

    input_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    with st.form("late_delivery_prediction_form"):
        columns = st.columns(2)
        order_attributes: dict[str, str] = {}
        for index, feature in enumerate(input_features):
            with columns[index % 2]:
                order_attributes[feature] = st.text_input(
                    feature,
                    key=f"late_delivery_input_{index}",
                )
        warning_threshold = st.number_input(
            "Prototype warning threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_WARNING_THRESHOLD,
            step=0.05,
            format="%.2f",
            key="late_delivery_warning_threshold",
        )
        submitted = st.form_submit_button(
            "Predict late-delivery probability",
            key="late_delivery_prediction_button",
        )

    if not submitted:
        return

    try:
        prediction = predict_late_delivery(
            order_attributes,
            warning_threshold=warning_threshold,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Late-delivery prediction unavailable: {exc}")
        return

    st.metric(
        "Predicted Late_delivery_risk probability",
        f"{prediction.probability:.1%}",
    )
    if prediction.warning:
        st.warning(
            "Late-delivery estimate meets or exceeds the selected "
            f"{prediction.warning_threshold:.2f} prototype threshold."
        )
    else:
        st.info(
            "Late-delivery estimate is below the selected "
            f"{prediction.warning_threshold:.2f} prototype threshold."
        )


def _render_results(
    result: DecisionResult,
    currency: str,
    quantity_unit: str,
) -> None:
    """Display recommendation outcome and all evaluated scenarios."""

    st.subheader("Incident context (informational only)")
    incident_context = result.incident_context
    if incident_context is None:
        st.write("No incident risk context was supplied for this evaluation.")
    else:
        if (
            incident_context.association_status
            is IncidentAssociationStatus.USER_CONFIRMED
        ):
            st.write(
                "Association: user-confirmed for this evaluation; "
                "not verified by a project-data link."
            )
        else:
            st.write("Association: unlinked; not confirmed for this evaluation.")

        assessment = incident_context.risk_assessment
        if assessment is None or assessment.score is None:
            st.warning("Risk index: Unassessed; do not interpret this as low risk.")
        else:
            st.write(
                f"Risk index: {assessment.score}/100 - "
                f"{assessment.level.value} (heuristic)"
            )
        if assessment is not None:
            for note in assessment.notes:
                st.caption(note)
    st.caption(
        "Incident context is carried with the evaluation for traceability only; "
        "it does not affect supplier feasibility, costs, or recommendation ranking."
    )
    prediction = result.disruption_prediction
    st.warning(
        "Operational disruption prediction: "
        f"{prediction.status.value.capitalize()}."
    )
    st.write(prediction.reason)
    st.caption(
        "This status is separate from event-category classification and the "
        "heuristic incident risk index."
    )
    with st.expander("Data required to enable operational prediction"):
        for requirement in prediction.required_data:
            st.markdown(f"- {requirement}")
    if (
        result.recommendation_status
        is RecommendationStatus.RISK_COMPARISON_UNAVAILABLE
    ):
        st.warning(
            "Risk-aware comparison unavailable; no risk-adjusted winner was selected."
        )
        if not result.has_feasible_scenario:
            st.warning(
                "No supplier scenario is confirmed feasible under the supplied "
                "constraints."
            )
        st.write("Scenario risk contribution: unavailable (not comparable).")
        for reason in result.risk_comparison_reasons:
            st.caption(reason)

    st.subheader("Recommendation")
    recommendation = result.recommendation
    if (
        result.recommendation_status
        is RecommendationStatus.COST_ONLY_RECOMMENDATION
    ):
        st.success(
            f"Recommended scenario: {recommendation.scenario_id} "
            f"({recommendation.action})"
        )
        st.write(
            "**Recommended allocations:** "
            + ", ".join(
                f"{allocation.supplier_id}: {allocation.quantity} {quantity_unit}"
                for allocation in recommendation.allocations
            )
        )
        if recommendation.cost is not None:
            st.write(
                "**Recommended total estimated cost:** "
                + _money(recommendation.cost.total_cost, currency)
            )
        st.write(recommendation.explanation)
    elif (
        result.recommendation_status
        is RecommendationStatus.NO_FEASIBLE_SCENARIO
    ):
        st.warning("No feasible supplier scenario satisfies the supplied constraints.")
        st.write(recommendation.explanation)
    elif (
        result.recommendation_status
        is RecommendationStatus.NO_COMPLETE_COST_SCENARIO
    ):
        st.warning(
            "Feasible scenarios exist, but none has a complete cost estimate; "
            "no cost-only recommendation was made."
        )
        st.write(recommendation.explanation)
    elif (
        result.recommendation_status
        is RecommendationStatus.NO_ELIGIBLE_SCENARIO
    ):
        st.warning(
            "No scenario is confirmed feasible with a complete cost estimate; "
            "no cost-only recommendation was made."
        )
        st.write(recommendation.explanation)
    elif result.recommendation_status is RecommendationStatus.UNSPECIFIED:
        st.warning(
            "Recommendation status is unspecified/unconfirmed; no scenario "
            "is presented as recommended."
        )
        st.write(recommendation.explanation)

    st.caption("Selection rule: " + recommendation.selection_rule)
    st.subheader("Scenario comparison")
    recommended_scenario_id = (
        recommendation.scenario_id
        if result.recommendation_status
        is RecommendationStatus.COST_ONLY_RECOMMENDATION
        else None
    )
    for scenario in result.scenarios:
        _render_scenario(
            scenario,
            currency,
            quantity_unit,
            recommended_scenario_id,
        )

    with st.expander("Assumptions and limitations"):
        st.markdown("**Assumptions**")
        for assumption in result.assumptions:
            st.markdown(f"- {assumption}")
        st.markdown("**Limitations**")
        for limitation in result.limitations:
            st.markdown(f"- {limitation}")


def main() -> None:
    """Render the interactive Streamlit dashboard."""

    st.set_page_config(
        page_title="Adaptive Supply-Chain Decision Engine",
        page_icon="📦",
        layout="wide",
    )
    st.title("Adaptive Supply-Chain Decision Engine")
    st.warning(
        "Cost-only is the default recommendation policy. Risk-aware mode "
        "currently withholds a winner because comparable scenario risk inputs "
        "are unavailable. The incident index is a heuristic, not a disruption "
        "probability."
    )
    st.info(
        "No operational data is preloaded. Values entered below are "
        "user-provided/demo inputs and are not verified project or supplier data."
    )

    st.subheader("Incident Analysis (NLP)")
    st.caption(
        "Classifies the incident text into an event category, then optionally "
        "shows a configurable heuristic index using the selected severity. "
        "The index is not validated or probabilistic and does not affect "
        "supplier recommendations."
    )
    with st.form("incident_analysis_form"):
        incident_description = st.text_area(
            "Incident description",
            key="incident_description",
            placeholder="Describe the reported supply-chain incident",
        )
        incident_severity = st.selectbox(
            "Incident severity",
            options=("Not provided", "Minor", "Moderate", "Severe", "Extreme"),
            key="incident_severity",
        )
        classify_submitted = st.form_submit_button(
            "Classify incident",
            key="classify_incident_button",
        )

    if classify_submitted:
        st.session_state["incident_analysis_result"] = None
        st.session_state["incident_risk_assessment"] = None
        st.session_state["incident_analysis_error"] = None
        st.session_state["confirm_incident_order_association"] = False
        try:
            event_category = classify_incident(
                incident_description
            )
            st.session_state["incident_analysis_result"] = event_category
            severity_value = (
                None if incident_severity == "Not provided" else incident_severity
            )
            st.session_state["incident_risk_assessment"] = score_incident(
                event_category,
                severity_value,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            st.session_state["incident_analysis_error"] = str(exc)

    incident_error = st.session_state.get("incident_analysis_error")
    incident_result = st.session_state.get("incident_analysis_result")
    if incident_error:
        st.error(incident_error)
    elif incident_result:
        st.success(f"Predicted event category: {incident_result}")
        risk_assessment = st.session_state.get("incident_risk_assessment")
        if isinstance(risk_assessment, RiskAssessment):
            if risk_assessment.score is None:
                st.info(
                    f"Heuristic incident risk index: "
                    f"{risk_assessment.level.value}"
                )
            else:
                st.info(
                    f"Heuristic incident risk index: {risk_assessment.score}/100 "
                    f"- {risk_assessment.level.value}"
                )
            for note in risk_assessment.notes:
                st.caption(note)

    _render_late_delivery_prediction()

    with st.sidebar:
        st.header("Scenario setup")
        alternative_count = st.number_input(
            "Number of alternative suppliers",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
        )

    incident_risk_assessment = st.session_state.get("incident_risk_assessment")
    with st.form("decision_engine_form"):
        st.subheader("Order")
        order_col1, order_col2 = st.columns(2)
        with order_col1:
            product_id = st.text_input(
                "Product ID",
                placeholder="Enter product ID",
                key="decision_product_id",
            )
            order_quantity = st.text_input(
                "Order quantity",
                placeholder="Enter a positive quantity",
                key="decision_order_quantity",
            )
            quantity_unit = st.text_input(
                "Quantity unit",
                placeholder="For example, units or kg",
                key="decision_quantity_unit",
            )
        with order_col2:
            currency = st.text_input(
                "Currency",
                value="USD",
                help="Enter the currency used for every cost, e.g. USD.",
            )
            st.caption(
                "Supplier capacities and allocations use the order quantity unit."
            )

        st.subheader("Incident association (optional)")
        if isinstance(incident_risk_assessment, RiskAssessment):
            incident_association_confirmed = st.checkbox(
                "I confirm this analyzed incident is associated with this order evaluation",
                key="confirm_incident_order_association",
                help=(
                    "This is a user assertion for this evaluation only. "
                    "The project data does not verify an incident-to-order link."
                ),
            )
        else:
            incident_association_confirmed = False
            st.caption(
                "No incident risk assessment is available. "
                "The decision engine will treat incident risk as unassessed."
            )

        decision_policy = st.selectbox(
            "Recommendation policy",
            options=list(DecisionPolicy),
            format_func=lambda policy: (
                "Cost-only (default)"
                if policy is DecisionPolicy.COST_ONLY
                else "Risk-aware (currently unavailable)"
            ),
            key="decision_policy",
            help=(
                "Risk-aware mode withholds a winner until comparable "
                "supplier-specific risk inputs and an approved conversion "
                "policy exist."
            ),
        )

        st.subheader("Current supplier")
        current_supplier_data = _supplier_fields(
            "Current supplier",
            "current",
            switching_cost_applicable=False,
        )

        st.subheader("Alternative suppliers")
        alternative_supplier_data = []
        for index in range(int(alternative_count)):
            with st.expander(
                f"Alternative supplier {index + 1}",
                expanded=index == 0,
            ):
                alternative_supplier_data.append(
                    _supplier_fields(
                        f"Alternative supplier {index + 1}",
                        f"alternative_{index}",
                        switching_cost_applicable=True,
                    )
                )

        st.subheader("Optional constraints")
        max_lead_time_enabled = st.checkbox("Apply maximum lead time")
        max_lead_time = st.text_input(
            "Maximum lead time (days)",
            placeholder="Enter limit",
            disabled=not max_lead_time_enabled,
        )
        max_budget_enabled = st.checkbox(
            "Apply maximum budget",
            key="max_budget_enabled",
        )
        max_budget = st.text_input(
            "Maximum budget (in selected currency)",
            key="max_budget",
            placeholder="Enter limit",
            disabled=not max_budget_enabled,
        )
        approved_suppliers_enabled = st.checkbox(
            "Restrict to approved suppliers"
        )
        approved_supplier_ids_text = st.text_input(
            "Approved supplier IDs (comma-separated)",
            placeholder="Enter supplier IDs",
            disabled=not approved_suppliers_enabled,
        )
        max_suppliers_enabled = st.checkbox(
            "Apply maximum number of suppliers per order"
        )
        max_suppliers = st.number_input(
            "Maximum suppliers per order",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            disabled=not max_suppliers_enabled,
        )

        enable_split_evaluation = st.checkbox(
            "Enable split-order evaluation",
            value=False,
            key="enable_split_evaluation",
        )
        split_values: list[str] = []
        if enable_split_evaluation:
            st.caption(
                "Enter a positive allocation for each supplier to include. "
                "The allocations must sum exactly to the order quantity."
            )
            all_supplier_data = [
                current_supplier_data,
                *alternative_supplier_data,
            ]
            for index, supplier_data in enumerate(all_supplier_data):
                supplier_label = supplier_data.supplier_id.strip() or supplier_data.label
                split_values.append(
                    st.text_input(
                        f"Split quantity - {supplier_label}",
                        key=f"split_{index}",
                        placeholder="Leave blank if not used",
                    )
                )

        submitted = st.form_submit_button(
            "Evaluate scenarios",
            type="primary",
            use_container_width=True,
            key="evaluate_scenarios_button",
        )

    if submitted:
        try:
            evaluation = _build_evaluation(
                product_id,
                order_quantity,
                quantity_unit,
                currency,
                current_supplier_data,
                alternative_supplier_data,
                max_lead_time_enabled=max_lead_time_enabled,
                max_lead_time=max_lead_time,
                max_budget_enabled=max_budget_enabled,
                max_budget=max_budget,
                approved_suppliers_enabled=approved_suppliers_enabled,
                approved_supplier_ids_text=approved_supplier_ids_text,
                max_suppliers_enabled=max_suppliers_enabled,
                max_suppliers=int(max_suppliers),
                incident_context=_build_incident_risk_context(
                    incident_risk_assessment
                    if isinstance(incident_risk_assessment, RiskAssessment)
                    else None,
                    incident_association_confirmed,
                ),
            )
            supplier_ids = [
                evaluation.current_supplier.supplier_id,
                *(
                    supplier.supplier_id
                    for supplier in evaluation.alternative_suppliers
                ),
            ]
            explicit_splits = _build_split_allocations(
                enable_split_evaluation,
                supplier_ids,
                split_values,
            )
            result = evaluate_order(
                evaluation,
                split_allocations=explicit_splits,
                policy=decision_policy,
            )
        except (ValueError, InvalidOperation) as exc:
            st.error(f"Could not evaluate the entered data: {exc}")
            return

        _render_results(
            result,
            result.currency,
            result.quantity_unit,
        )


if __name__ == "__main__":
    main()
