# Adaptive Decision Engine

## What it does

This version includes a separate NLP incident-classification section and
compares sourcing scenarios for one order using operational values explicitly
supplied by the caller. It evaluates continuing with the current supplier,
switching the full order to each alternative, and any split-order allocations
explicitly supplied to `evaluate_order`.

The recommendation policy defaults to **cost-only**. An optional `risk_aware`
policy can be requested, but it currently returns **not comparable** and
withholds a winner; the project does not contain the supplier/scenario risk
inputs or approved risk-to-cost conversion needed for a defensible comparison.
The supplier decision policies do not consume late-delivery probability,
incident classification, or Person 1's NLP classifier confidence.

## Incident analysis (NLP)

The dashboard's **Incident Analysis (NLP)** section accepts an incident
description and uses Person 1's saved scikit-learn event-classification
pipeline at
`person1_nlp/models/supply_chain_event_pipeline.joblib`. The path is resolved
relative to `person3_decision/app.py`, and Streamlit caches the loaded model
resource. The pipeline returns an event category such as `Weather`; it does
not return disruption probability. The category and its derived heuristic
index do not affect supplier scenario costs or recommendation ranking.

For example, the synthetic/example text
`Heavy rainfall caused flooding and delayed transportation.` is expected to
be classified as `Weather` by the saved artifact. This label describes the
classifier output only; it is not a risk score or a claim about operational
impact.

## Late-delivery prediction

The dashboard also provides a standalone **Late-delivery prediction
(order-time inputs)** form. It reuses the saved
`person1_nlp/supply-chain-prediction/late_delivery_model.joblib` pipeline;
the artifact path is resolved from the source file, independent of the current
working directory. The raw input columns are passed directly to the saved
pipeline, which owns its fitted preprocessing:

- Numeric: scheduled shipment days, item quantity, item product price, item
  discount, item discount rate, latitude, and longitude.
- Categorical: payment type, shipping mode, market, order region, category
  name, department name, and customer segment.

All required inputs are validated before inference. Missing/invalid values,
artifact-loading errors, and prediction failures are shown in the dashboard
instead of terminating it. The displayed probability is for the dataset's
`Late_delivery_risk` target (which matches the `Delivery Status == "Late
delivery"` label); it is **not** a general supply-chain disruption probability
or a prediction about an incident or supplier.

The warning threshold defaults to **0.4** and can be changed in the form. It
is a selected prototype operating threshold, not a calibrated probability
cutoff or a proven business optimum. The late-delivery output remains separate
from NLP incident classification, the heuristic incident-risk index, supplier
risk estimates, and decision-engine inputs. It does not change supplier
rankings, costs, allocations, or the cost-only recommendation policy.

## Heuristic incident risk index

After classifying an incident, the dashboard can calculate a separate
heuristic index from the predicted event category and the user-selected
severity. The pure scoring interface is:

```python
from person3_decision.risk_scoring import score_incident

assessment = score_incident("Weather", "Severe")
print(assessment.score, assessment.level.value)
```

`RiskScoringConfig` in `person3_decision/risk_scoring.py` provides configurable
category points, severity points, a maximum score, and inclusive level
boundaries. The current defaults are provisional illustrative choices, not
derived from training outcomes or approved by an operational risk owner. For
complete, recognized inputs, the index adds category and severity points and
caps the result at 100. Missing or unrecognized inputs return `score=None` and
the `Unassessed` level instead of being treated as zero.

This is a heuristic index, **not a validated prediction or calibrated
probability of a future disruption**. The project data has event categories
and severity labels but no linked shipment/supplier outcome target. The index
is informational context only and does not alter the cost-only supplier
recommendation. Domain-owner approval of the mappings and thresholds,
followed by validation against outcome-linked records, is required before
operational use.

## Operational disruption prediction status

Operational disruption prediction is currently **unavailable**. The inspected
GMRID train/test schemas provide incident `id`, text, `Severity`, `Region`,
`Datetime`, `Category`, and `Summarized_label`. The classifier training scripts
use `Summarized_label` as the target, and the test-prediction export compares
`predicted_label` against that event-class label. These are event categories,
not observed future disruptions of particular orders, shipments, or suppliers.
The NLP feature exports include `predicted_event_type` and
`event_prediction_confidence`; that confidence applies only to the event-class
prediction and is not disruption probability.

Each `DecisionResult` carries a typed `disruption_prediction` with
`status="unavailable"`, `probability=None`, a reason, and the data prerequisites
to change that status. The dashboard and synthetic demo display this
availability separately from NLP classification and the heuristic index. It
does not affect cost-only ranking or enable risk-aware comparison.

Enabling supervised operational prediction requires, at minimum:

1. Stable order, shipment, and supplier IDs and an auditable relationship
   between incident records and the affected operational records.
2. A defined prediction-time point and a domain-approved future horizon.
3. Outcome labels for that unit and horizon, including non-disrupted examples,
   explicit outcome definitions, and event timestamps. Labels must be based on
   observed operational outcomes rather than inferred from incident category
   or severity.
4. For impact prediction, measured delay durations or financial losses with
   documented units, currency, and measurement windows.

Once available, establish a time-ordered, leakage-safe train/validation/test
split and evaluate against the approved target with suitable classification
metrics (including class-sensitive metrics for a binary disruption target) or
regression metrics for measured impact. No such target or evaluation is
implemented now.

## Incident-to-evaluation context contract

`IncidentRiskContext` is an optional typed field on `EvaluationInput` and is
carried unchanged into `DecisionResult`. It contains:

- `association_status`: `IncidentAssociationStatus.UNLINKED` or
  `IncidentAssociationStatus.USER_CONFIRMED`.
- `risk_assessment`: a `RiskAssessment` or `None`. A `RiskAssessment` may itself
  be unassessed (`score=None`, level `Unassessed`).

The dashboard presents an unchecked-by-default confirmation for the current
evaluation. Checking it means only that the user asserts the analyzed incident
is relevant to this evaluation; the repository has no verified incident/order
join key, and the current order form has no unique order ID. Without
confirmation, an available assessment is carried as `UNLINKED`. With no
assessment, the context is absent. Missing or unassessed risk is displayed
explicitly and must not be interpreted as low risk.

This contract is for traceability, not supplier-specific exposure. The
assessment is not copied to each supplier or scenario.

## Decision policies and result status

`evaluate_order(evaluation, split_allocations=(), policy=...)` accepts
`DecisionPolicy.COST_ONLY` (the default) or `DecisionPolicy.RISK_AWARE`.

- **Cost-only:** unchanged behavior. Among feasible scenarios with complete
  costs, select the lowest total of purchase cost plus applicable switching
  cost; exact ties use `scenario_id`. Infeasible/unknown feasibility and
  incomplete costs remain excluded as before.
- **Risk-aware:** currently blocked. It returns
  `RiskComparisonStatus.NOT_COMPARABLE`, an explicit list of missing
  prerequisites, and no recommendation. It does not silently fall back to a
  cost-only winner. Scenario feasibility and `CostEstimate` values remain
  available separately for inspection.
- `recommendation_status` makes the outcome explicit:
  `cost_only_recommendation`, `no_feasible_scenario`,
  `no_complete_cost_scenario`, `no_eligible_scenario`, or
  `risk_comparison_unavailable`. The last status means the requested policy
  could not compare risk and never exposes a cost-only winner as a risk-aware
  result. `has_feasible_scenario` reports whether any scenario is explicitly
  marked feasible, independently of recommendation status.
- `unspecified` is the compatibility default for manually constructed
  `DecisionResult` objects; `evaluate_order` always assigns an outcome status.
  Downstream consumers must treat it as unconfirmed regardless of whether a
  recommendation scenario ID is populated.
- `no_feasible_scenario` is used only when every evaluated scenario is
  explicitly infeasible. `no_eligible_scenario` covers cases where none is
  explicitly infeasible but no scenario qualifies for recommendation, such as
  unknown feasibility. `no_complete_cost_scenario` means at least one scenario
  is feasible, but none has a complete cost estimate.

The current engine has no supplier/scenario risk-input contract or
risk-contribution values. Therefore risk contribution is **unavailable**, not
zero. Unlinked incidents, missing assessments, and `Unassessed` scores cannot
be treated as low risk. Even a user-confirmed incident with a numeric
heuristic score is insufficient for risk-aware ranking.

No risk-adjusted objective formula or currency conversion has been selected
or implemented. A future policy must not combine money and heuristic points
until domain owners approve (1) supplier and allocation-specific exposure
inputs, (2) a risk-to-cost conversion or another comparable decision rule,
and (3) behavior for missing scenario values. Those decisions and the required
data are blockers; the provisional category/severity weights in
`risk_scoring.py` do not define them.

## Requirements and execution

- Python 3.10 or later (the repository's selected interpreter is Python 3.13).
- The decision engine and tests use only the standard library.
- The dashboard requires Streamlit, joblib, and the scikit-learn version
  compatible with the saved pipeline; these are available in the selected
  workspace environment.
- Risk scoring and its unit tests use only the Python standard library.

From the repository root, run the standard-library test suite with:

```powershell
& "C:\Users\Lakkshya\AppData\Local\Programs\Python\Python313\python.exe" -m unittest discover -s person3_decision/tests -v
```

Run the synthetic demonstration from the repository root with:

```powershell
& "C:\Users\Lakkshya\AppData\Local\Programs\Python\Python313\python.exe" -m person3_decision.demo
```

Launch the Streamlit dashboard from the repository root with:

```powershell
& "C:\Users\Lakkshya\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run person3_decision/app.py
```

The package can be imported from the repository root:

```python
from person3_decision import (
    EvaluationInput,
    OrderInput,
    SupplierOption,
    evaluate_order,
)
from person3_decision.decision_engine import Constraints
```

## Input schema

Call `evaluate_order(evaluation, split_allocations=())`.

`EvaluationInput` contains:

- `order: OrderInput`
  - `product_id: str`
  - `order_quantity: number` greater than zero
  - `quantity_unit: str`
  - `currency: str` (stored uppercase)
- `current_supplier: SupplierOption`
- `alternative_suppliers: tuple[SupplierOption, ...]`
- `constraints: Constraints` (optional; defaults to no constraints)

Each `SupplierOption` contains:

- `supplier_id: str`, unique within an evaluation
- `available_quantity: number`, non-negative and in the order quantity unit
- `lead_time_days: number`, non-negative
- `unit_cost: number | None`, non-negative in the declared currency per unit
- `switching_cost: number | None`, non-negative in the declared currency

For costs, `None` means unknown; a supplied numeric zero is a confirmed zero.
An unknown unit cost makes the scenario cost incomplete. A switching fee is
applicable once per used non-current supplier; if that fee is `None`, the
scenario's switching cost is unknown. The current supplier's switching fee is
not applied.

`Constraints` supports:

- `max_lead_time_days: number | None`
- `max_budget: number | None` in the declared currency
- `approved_supplier_ids: frozenset[str] | None`
- `max_suppliers_per_order: int | None`

`split_allocations` is a sequence of explicit mappings from supplier ID to
positive allocated quantity. Each split must reference known suppliers and
sum exactly to the order quantity. No split percentage is generated by the
engine.

Numbers may be supplied as integers, floats, numeric strings, or `Decimal`.
Internally, numeric values are converted to finite `Decimal` values. Invalid
or negative required operational values raise `ValueError`; incomplete cost
values are represented as unknown rather than defaulted to zero.

## Scenario and cost rules

Generated candidates are:

1. Full order with the current supplier.
2. Full order with each listed alternative supplier.
3. Each explicitly provided split allocation.

Capacity, approved-supplier, maximum-supplier-count, and lead-time
constraints are enforced. Infeasible candidates remain in the result with
reasons. If the maximum budget cannot be checked because cost is incomplete,
feasibility is `unknown`, not assumed feasible.

Purchase cost is the sum of allocated quantity times supplied unit cost.
Switching cost is the sum of supplied, applicable fees. Total cost includes
only those two components. It does not include transportation, inventory
holding, stockout, or disruption-loss costs. A missing required cost makes the
estimate incomplete and prevents it from being recommended.

## Output schema

`evaluate_order` returns a `DecisionResult` containing:

- Order/product identifiers, quantity, unit, and currency.
- `scenarios`: each has a scenario ID, action, allocations, feasibility
  (`feasible`, `infeasible`, or `unknown`), reasons, and cost estimate.
- Each cost estimate includes per-supplier and total purchase/switching costs,
  total cost, a `complete` flag, and unknown component names.
- `recommendation`: selected scenario ID/action/allocations/cost, or no
  selection (`None`), together with the selection rule and explanation.
- `recommendation_policy`: `cost_only` by default, or `risk_aware` when
  explicitly requested.
- `residual_risk`, always `not_assessed`.
- `risk_comparison_status`: `not_requested` for cost-only, or
  `not_comparable` for the currently blocked risk-aware request.
- `risk_comparison_reasons`: explicit prerequisites that prevent risk-aware
  comparison. No risk contribution is fabricated for any scenario.
- `recommendation_status`: the explicit outcome state listed above.
- `has_feasible_scenario`: whether any scenario is explicitly feasible.
- `disruption_prediction`: typed operational prediction availability. It is
  `unavailable` with `probability=None` until outcome-linked data and a
  prediction contract are established.
- Assumptions and limitations.

Under `cost_only`, only scenarios that are feasible and have complete costs
are ranked. The lowest total cost wins; exact cost ties are broken by
lexicographically smallest scenario ID. No eligible scenario means no
recommendation. Under `risk_aware`, no scenario is ranked until the missing
supplier/scenario risk-input and approved conversion contracts are implemented.

## Synthetic example

The following values are synthetic examples for demonstrating the interface;
they are not real project or supplier data.

```python
from person3_decision import (
    EvaluationInput,
    OrderInput,
    SupplierOption,
    evaluate_order,
)
from person3_decision.decision_engine import Constraints

example = EvaluationInput(
    order=OrderInput(
        product_id="SYNTHETIC-PRODUCT",
        order_quantity=10,
        quantity_unit="units",
        currency="USD",
    ),
    current_supplier=SupplierOption(
        supplier_id="SYNTHETIC-CURRENT",
        available_quantity=20,
        lead_time_days=5,
        unit_cost=10,
    ),
    alternative_suppliers=(
        SupplierOption(
            supplier_id="SYNTHETIC-ALTERNATIVE",
            available_quantity=20,
            lead_time_days=4,
            unit_cost=8,
            switching_cost=3,
        ),
    ),
    constraints=Constraints(max_lead_time_days=7),
)

result = evaluate_order(example)
print(result.recommendation.scenario_id)
```

## Limitations and future integration

- The NLP model classifies incident text into event categories only. The
  separately displayed heuristic index uses category and user-provided
  severity but is not validated, probabilistic, or evidence of actual
  supplier impact. Neither changes the cost-only supplier recommendation.
- The dashboard requires the saved model artifact at the documented relative
  path. Missing or unreadable artifacts and inference failures are displayed
  to the user; the model is not retrained by the dashboard.
- The repository currently has no operational supplier, order, inventory,
  demand, route, or cost dataset. Real recommendations require validated
  values supplied to this interface.
- Feasibility is only relative to the input values and constraints; it is not
  a guarantee about real-world fulfillment.
- The current policy ranks direct, known costs only. It does not claim that a
  selected scenario reduces disruption risk.
- Person 1's `event_prediction_confidence` is classifier confidence for an
  event category, not disruption probability, and is not used here.
- Person 2's model interface and output semantics are not present in the
  repository. A future integration can add a separately typed prediction
  input after its target, horizon, entity mapping, and value meaning are
  confirmed. Until then, no Person 2 output is assumed.
- Safety-stock and transport-route scenarios need additional operational
  inputs and cost rules before they can be evaluated responsibly.
