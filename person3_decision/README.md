#  Adaptive Decision Engine

## Objective
Build a decision-support module that recommends suitable actions when supply-chain disruptions are predicted.

## Planned modules
- `scenarios.py` — defines alternative response strategies.
- `cost_model.py` — estimates the cost of each strategy.
- `recommendation.py` — ranks strategies and explains the selected action.
- `app.py` — provides an interactive dashboard.
- `tests/` — tests the decision engine.

## Initial response strategies
1. Continue with the current supplier.
2. Switch to an alternative supplier.
3. Split the order across suppliers.
4. Increase safety stock.
5. Change the transportation route.

## Integration
The module will use predictions from the ML component and relevant features from the NLP component.

**Note:** Initial cost estimates and example predictions are illustrative until validated data and model outputs are integrated.
