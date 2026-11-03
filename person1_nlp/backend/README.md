# FastAPI backend

This backend serves only the existing trained models. It does not implement supplier recommendations, incident severity/risk heuristics, or general disruption probability.

## Installation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

Dependencies are pinned in `backend/requirements.txt`; Pydantic 2.12.5 is selected because it provides a Python 3.14-compatible wheel in the audit environment. The model artifacts were audited with pandas 3.0.6, scikit-learn 1.9.1, and joblib 1.6.0.

## Model files

Defaults are resolved from the repository root:

- NLP: `models/supply_chain_event_pipeline.joblib`
- Delivery: `supply-chain-prediction/late_delivery_model.joblib`

Both models are loaded once during FastAPI startup. Missing or incompatible models cause the health endpoint to report `degraded` and the affected prediction endpoint to return HTTP 503. No dummy prediction is used.

## Environment variables

- `NLP_MODEL_PATH`: optional path to the NLP joblib artifact.
- `DELIVERY_MODEL_PATH`: optional path to the late-delivery joblib artifact.
- `DELIVERY_THRESHOLD`: optional validated configuration value between 0 and 1.
- `CORS_ORIGINS`: comma-separated origins; defaults to `http://localhost:3000`.

## Start and test

From the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload
pytest backend/tests
```

## Endpoints

### `GET /api/health`

```json
{
  "status": "ok",
  "models": {
    "nlp_model_loaded": true,
    "delivery_model_loaded": true
  }
}
```

### `POST /api/incidents/classify`

Request:

```json
{
  "description": "Heavy rainfall caused flooding and delayed transportation."
}
```

Response:

```json
{
  "predicted_category": "Weather",
  "model_confidence": 0.84,
  "confidence_message": "Model confidence is confidence in the predicted incident category; it is not disruption probability."
}
```

### `POST /api/delivery/predict`

The request uses the exact 14 model feature names, plus optional `threshold`:

```json
{
  "Days for shipment (scheduled)": 2,
  "Order Item Quantity": 2,
  "Order Item Product Price": 100.0,
  "Order Item Discount": 10.0,
  "Order Item Discount Rate": 0.1,
  "Latitude": 12.9716,
  "Longitude": 77.5946,
  "Type": "DEBIT",
  "Shipping Mode": "Standard Class",
  "Market": "Pacific Asia",
  "Order Region": "Southeast Asia",
  "Category Name": "Cleats",
  "Department Name": "Footwear",
  "Customer Segment": "Consumer",
  "threshold": 0.4
}
```

Response:

```json
{
  "estimated_late_delivery_probability": 0.5631,
  "prediction": 1,
  "warning": true,
  "threshold_used": 0.4,
  "target": "Late_delivery_risk"
}
```

The probability is the existing model's estimate for the existing `Late_delivery_risk` target. It is not a general disruption probability.
