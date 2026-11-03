# Migration Audit

Audit date: 2026-11-01  
Repository: `person1_nlp`  
Audit scope: read-only inspection of the repository, serialized models, datasets, and executable scripts.

## Executive summary

This repository is a collection of Python data-preparation and model-training/prediction scripts. It is not currently a Streamlit application and does not contain a FastAPI backend, supplier decision engine, supplier master data, incident-risk calculation service, general disruption predictor, frontend, dependency manifest, README, or automated test suite.

Implemented capabilities are:

- GMRID incident text cleaning and supervised incident-category classification.
- Export of incident predictions and Logistic Regression confidence values.
- A scikit-learn pipeline for `Late_delivery_risk` classification from order-time features.
- One hard-coded late-delivery prediction demonstration.

The incident classifier's confidence is a class-model confidence, not a probability that a disruption will occur. The repository contains no implementation that converts incident severity into a validated risk probability or risk index. Supplier recommendations cannot be migrated faithfully yet because the required decision engine, supplier records, and constraint definitions are absent.

No existing project file was modified by this audit. This report is the only new project file.

## 1. Existing architecture

```text
Raw GMRID CSVs
  -> src/preprocess_text.py
  -> data/processed/gmrid_*_processed.csv
  -> src/train_classifier.py
       -> models/gmrid_classifier.joblib
       -> models/tfidf_vectorizer.joblib
       -> data/processed/gmrid_test_predictions.csv
  -> src/export_nlp_features.py
       -> data/processed/nlp_events_{train,test}.csv

DataCoSupplyChainDataset.csv
  -> supply-chain-prediction/train_*.py
       -> late_delivery_model.joblib (only some training scripts save it)
  -> supply-chain-prediction/predict_late_delivery.py
```

There is no application layer connecting these artifacts. The two model families are separate and there is no current API or UI integration point.

Repository inventory:

- `src/`: GMRID download, inspection, preprocessing, training, export, and a small NLP smoke script.
- `models/`: two separate NLP representations plus the combined pipeline.
- `data/raw/`: GMRID and auxiliary CSVs.
- `data/processed/`: processed GMRID data, stored predictions, and exported NLP features.
- `supply-chain-prediction/`: DataCo dataset, late-delivery model training experiments, and a prediction demo.
- Root scripts: model comparison and pipeline serialization.
- No `app.py`, `streamlit_app.py`, `pages/`, `api/`, `backend/`, `frontend/`, `package.json`, `requirements.txt`, `pyproject.toml`, `README.md`, or test directory was found.

## 2. Existing functionality

### NLP incident classification

Implementation files:

- `src/preprocess_text.py`: reads `Headline` and `Details`, combines them, lowercases, removes URLs and HTML, replaces non-ASCII letters/non-letters with spaces, and writes processed CSVs.
- `src/train_classifier.py`: fits a `TfidfVectorizer` and balanced `LogisticRegression`; filters missing labels, empty text, and test labels unseen in training; saves the classifier, vectorizer, and test predictions.
- `save_pipeline.py`: fits and saves an equivalent end-to-end `Pipeline` containing TF-IDF and Logistic Regression.
- `compare_models.py`: trains temporary Logistic Regression and Linear SVM models and writes a comparison CSV.
- `src/export_nlp_features.py`: loads the separately serialized classifier/vectorizer and exports predicted event type and confidence alongside metadata.
- `models/supply_chain_event_pipeline.joblib`: preferred self-contained NLP artifact for serving because preprocessing and classifier are coupled.

Inputs:

- Training data columns: `clean_text`, `Summarized_label` plus metadata.
- Runtime input to the serialized pipeline: a string or list of strings. The serving code must use the same cleaning convention as training; the saved pipeline itself only vectorizes and does not perform the repository's `clean_text` function.

Outputs:

- One of eight labels: `Accident`, `Administrative Issue`, `Cyber Attack`, `Human Error`, `Others`, `Terrorism`, `Weather`, or `Worker Strike`.
- `predict_proba` maximum is exported as `event_prediction_confidence`. This is not disruption probability and is not a calibrated severity score.

Observed stored test result from `data/processed/gmrid_test_predictions.csv`:

- 1,147 rows.
- Accuracy: 0.911072.
- Macro-F1: 0.796010.
- Weighted-F1: 0.910252.

These metrics describe the stored predictions and are not a newly rerun evaluation of a clean training process.

### Incident severity and risk

The GMRID data has a `Severity` field with values `Minor`, `Moderate`, `Severe`, and `Extreme`. The repository only carries this field through preprocessing/export and displays its distribution. No severity scoring function, heuristic risk-index formula, incident aggregation, calibration, or probability validation was found.

Therefore a future API must label any derived severity/risk output explicitly as heuristic and must not present it as a validated probability. At present this feature is missing, not partially implemented.

### Late-delivery model

Implementation files:

- `supply-chain-prediction/train_model.py`: stratified random 80/20 split; trains a preprocessing-plus-Logistic-Regression pipeline but does not save it.
- `supply-chain-prediction/train_grouped_model.py`: order-grouped split using `Order Id`; evaluates but does not save.
- `supply-chain-prediction/train_chronological_model.py`: trains before 2017 and evaluates on 2017 onward; saves `late_delivery_model.joblib` using a path relative to the current working directory.
- `supply-chain-prediction/predict_late_delivery.py`: loads the neighboring joblib, predicts the positive class probability for one hard-coded sample, and applies threshold `0.4`.
- `supply-chain-prediction/late_delivery_model.joblib`: serialized pipeline currently used by the demo.

Input features, in exact model order:

- Numeric: `Days for shipment (scheduled)`, `Order Item Quantity`, `Order Item Product Price`, `Order Item Discount`, `Order Item Discount Rate`, `Latitude`, `Longitude`.
- Categorical: `Type`, `Shipping Mode`, `Market`, `Order Region`, `Category Name`, `Department Name`, `Customer Segment`.

Target and output:

- Target is the existing `Late_delivery_risk` field, binary classes `[0, 1]`.
- The demo returns a class-1 probability and thresholded warning. It does not predict a newly defined target.
- The full source dataset contains 180,519 rows and the target distribution is approximately 45.17% class 0 / 54.83% class 1.

Runtime check:

- `predict_late_delivery.py` completed successfully on the audit environment and returned 56.31% for its hard-coded example, classified as a warning at threshold 0.4.

### Supplier decision engine

No supplier decision engine was found. No classes/functions, supplier inputs, cost constraints, capacity constraints, budget handling, lead-time constraints, objective function, ranking logic, or supplier output schema exists in the repository. There is also no supplier-risk dataset or model.

The migration must not invent supplier-risk information. A recommendation endpoint should either remain unavailable or require an explicit supplier catalog containing the fields needed by the existing business rules once those rules are supplied. It should not infer supplier risk from the incident classifier or late-delivery probability without an authorized design and data linkage.

### General disruption prediction

No general disruption prediction model or API was found. GMRID classification predicts an incident category for supplied text; it does not predict whether a future disruption will occur. This capability must not be shown as implemented in the new UI.

### Existing APIs and UI

- No FastAPI, Flask, Streamlit, route, request schema, or server entry point was found.
- `src/download_news.py` calls the external GDELT API to download recent articles, but this is a batch script, not an application API, and writes `data/raw/news.csv`.
- `src/download_gmrid.py` downloads datasets from Hugging Face, also as a batch script.

## 3. File-by-file component mapping

| File | Role | Main functions/classes | Inputs/outputs | Integration status / issues |
|---|---|---|---|---|
| `src/preprocess_text.py` | GMRID preprocessing | `clean_text` | Raw GMRID CSV -> processed CSV | Top-level execution; writes on import/run; ASCII-only cleaning may discard non-English characters |
| `src/train_classifier.py` | NLP training/evaluation | top-level pipeline | Processed CSV -> classifier, vectorizer, predictions | Top-level execution; no reusable service function |
| `src/export_nlp_features.py` | NLP feature export | `generate_nlp_features` | Processed data + model -> feature CSV | Reusable function is embedded in a script that executes and writes at import/run |
| `save_pipeline.py` | Combined NLP artifact training | top-level pipeline | Processed train CSV -> `supply_chain_event_pipeline.joblib` | Saves artifact; only self-contained NLP model |
| `compare_models.py` | Offline comparison | top-level experiment | Processed CSV -> comparison CSV | Writes output during experiment; not serving code |
| `src/test_nlp.py` | Basic smoke/demo script | local `clean_text` | Four literals -> `nlp_test_output.csv` | Not an automated test; imports unused `nltk`; currently fails before execution |
| `src/download_gmrid.py` | Dataset download | top-level request loop | Hugging Face -> raw CSVs | External network dependency; writes files |
| `src/download_news.py` | News download | top-level request | GDELT -> `data/raw/news.csv` | External API/rate-limit dependency; no classifier integration |
| `src/inspect_data.py` | Auxiliary CSV inspection | top-level | Raw auxiliary CSVs -> console | Requires files; no application integration |
| `src/inspect_gmrid.py` | GMRID inspection | top-level | Raw GMRID CSVs -> console | Offline diagnostic only |
| `supply-chain-prediction/train_model.py` | Random-split late model experiment | top-level | DataCo -> metrics | Does not save model; potential leakage across order items is why grouped/chronological variants exist |
| `supply-chain-prediction/train_grouped_model.py` | Grouped late model experiment | top-level | DataCo grouped split -> metrics | Does not save model |
| `supply-chain-prediction/train_chronological_model.py` | Chronological late model training | top-level | DataCo -> metrics + joblib | Saves relative to current directory, so invocation directory changes artifact location |
| `supply-chain-prediction/predict_late_delivery.py` | Late model demo | top-level | Hard-coded order -> console | No request interface; hard-coded threshold and sample |
| `supply-chain-prediction/analyze_target.py` | Target analysis | top-level | DataCo -> console | Requires working directory to contain dataset |
| `supply-chain-prediction/check_dates.py` | Date diagnostic | top-level | DataCo -> console | Same relative-path dependency |
| `supply-chain-prediction/check_duplicates.py` | Duplicate/order diagnostic | top-level | DataCo -> console | Same relative-path dependency |
| `supply-chain-prediction/check_target.py` | Target diagnostic | top-level | DataCo -> console | Same relative-path dependency |
| `supply-chain-prediction/inspect_dataset.py` | Dataset inspection | top-level | DataCo -> console | Same relative-path dependency |
| `models/*.joblib` | Serialized model artifacts | sklearn objects | Runtime model inputs -> predictions | Version-sensitive pickle/joblib format; no manifest of training versions/features |

## 4. Test and verification results

No `pytest`/`unittest` test suite or integration-test directory exists. The audit did not treat training scripts as tests.

Commands/checks performed:

1. Python AST parsing of all 19 Python files: passed; no syntax errors.
2. `src/test_nlp.py`: failed before its smoke logic with `ModuleNotFoundError: No module named 'nltk'`. `nltk` is imported but unused.
3. `supply-chain-prediction/predict_late_delivery.py`: passed; probability 56.31%, warning at threshold 0.4.
4. Serialized artifact loading and schema inspection: passed for all four joblib files.
5. Stored NLP prediction metrics: 1,147 rows, accuracy 0.911072, macro-F1 0.796010, weighted-F1 0.910252.

The environment reported Python 3.14.7, pandas 3.0.6, scikit-learn 1.9.1, and joblib 1.6.0; `nltk` was unavailable. These versions were observed at audit time, not declared by the project.

## 5. Bugs, gaps, and compatibility problems

- There is no current Streamlit entry point to migrate.
- There are no declared Python dependencies, so deployment cannot be reproduced reliably.
- There are no frontend files or Node dependency files.
- There is no automated regression test coverage.
- Most scripts execute work at module import time, which is unsafe for a web worker and makes testing difficult.
- The self-contained NLP pipeline does not include the repository's `clean_text` function; serving raw headline/details would not exactly reproduce training preprocessing unless cleaning is explicitly applied before `pipeline.predict`.
- `gmrid_classifier.joblib` and `tfidf_vectorizer.joblib` are separate artifacts; loading the wrong pair could silently produce incompatible features. The combined pipeline is safer.
- `event_prediction_confidence` is model confidence, not a disruption probability or validated risk score.
- `Severity` is passed through as a label; no heuristic risk formula is present.
- The late model's threshold 0.4 is described as an initial/selected warning threshold, but no calibration, cost curve, or persisted threshold configuration exists.
- The late-model training variants disagree on validation protocol; the selected production artifact's provenance is not recorded in metadata.
- `train_model.py` and `train_grouped_model.py` do not save their trained artifacts. `train_chronological_model.py` saves to a current-working-directory-relative path.
- Several DataCo scripts rely on `DataCoSupplyChainDataset.csv` in the process working directory rather than resolving from the script path.
- `src/test_nlp.py` is named like a test but writes an output file and is not discoverable as a test method; its unused `nltk` import blocks it.
- The dataset contains highly missing columns such as `Product Description` and `Order Zipcode`, although those are not currently model features.
- Joblib/scikit-learn artifacts are pickle-based and need pinned, compatible runtime versions and a trusted-artifact policy.
- External downloads have network, availability, rate-limit, and data-drift dependencies and are not behind an API abstraction.
- No authentication, authorization, input limits, logging, model versioning, health checks, or error contract exists because there is no backend.

## 6. Proposed FastAPI endpoint design

These are design proposals, not existing endpoints.

### Health and model metadata

- `GET /health` -> service status and loaded model status.
- `GET /models` -> model names, artifact versions, training/validation metadata, feature schema, and explicit capability flags.

### Incident classification

- `POST /v1/incidents/classify`

Request:

```json
{
  "text": "Heavy rainfall caused flooding and delayed transportation.",
  "headline": null,
  "details": null
}
```

Response should include `incident_category`, `classification_confidence`, `model_version`, and a statement that confidence is class confidence. If `headline`/`details` are accepted, combine and clean them with the training-compatible function; define precedence when both `text` and component fields are supplied.

### Incident severity/risk

- `POST /v1/incidents/assess`

Keep this endpoint disabled or return an explicit `not_implemented` capability until the heuristic formula is supplied and tested. If implemented later, return separate fields such as `severity_label`, `heuristic_risk_index`, `risk_scale`, and `is_probability: false`; never call it a probability.

### Late-delivery prediction

- `POST /v1/delivery-risk/predict`

Request contains the 14 exact model features listed above. Response should include `late_delivery_risk_probability`, `late_delivery_risk_prediction`, `threshold`, `target: "Late_delivery_risk"`, model version, and validation metadata. Pydantic validation should reject missing required fields or normalize only documented numeric/categorical forms.

### Supplier recommendations

- Do not expose `POST /v1/suppliers/recommend` as an active endpoint until the existing engine and supplier data are provided.
- Once available, the request should include order requirements, budget, capacity/quantity, lead-time constraints, and an explicit risk-data availability indicator.
- The response should show selected suppliers, feasible alternatives, constraint checks, costs, capacities, lead times, and the source/version of any supplier-risk signal. Missing risk information must be returned as missing, not fabricated or imputed silently.

### Capability signaling

Expose capability flags in `/models` or a dashboard configuration: incident classification available; late-delivery prediction available; heuristic severity/risk unavailable; supplier optimization unavailable; general disruption prediction unavailable.

## 7. Proposed Next.js page structure

Proposed Next.js 15 app-router pages:

- `/` — capability-aware overview dashboard; show only implemented model cards and data provenance.
- `/incidents/classify` — incident text form, predicted category, class confidence disclaimer, and input text.
- `/delivery-risk` — 14-field order form, prediction, threshold, target name, and model metadata.
- `/suppliers` — disabled/coming-soon state until the decision engine and supplier data are integrated; do not render invented recommendations.
- `/models` — model versions, feature schemas, validation metrics, data dates, and capability status.
- `/about` — system scope, limitations, and distinction between classification confidence, heuristic risk, and late-delivery model probability.

Shared components should include validated forms, loading/error states, capability badges, model provenance cards, a disclaimer component, and Recharts visualizations only for returned data. A chart must not imply general disruption forecasts or supplier risk where no such data exists.

## 8. Migration risks

1. **Scope mismatch:** the requested legacy Streamlit and supplier engine are absent, so a direct feature-preserving migration is currently impossible.
2. **Model semantics:** incident class confidence could be mislabeled as disruption probability or risk.
3. **Preprocessing drift:** frontend/backend input transformations could differ from the training scripts.
4. **Artifact compatibility:** joblib files need pinned Python/scikit-learn/joblib versions and checksums/version metadata.
5. **Validation ambiguity:** three late-model training protocols exist, but the deployed artifact's training protocol is not encoded.
6. **Business-rule loss:** supplier cost/capacity/budget/lead-time rules cannot be reproduced without their source implementation and data.
7. **Data privacy/security:** the DataCo CSV includes customer and order fields; a web API must minimize returned/stored data and protect model inputs/logs.
8. **Operational behavior:** current scripts are batch-oriented, write files, print to stdout, and have no concurrency or failure contract.
9. **Threshold governance:** the 0.4 warning threshold needs owner, calibration evidence, and versioning.
10. **No regression baseline:** there are no automated tests to protect parity during extraction into import-safe backend modules.

## 9. Recommended implementation sequence

1. Obtain the missing Streamlit entry point, supplier decision engine, supplier catalog/schema, and any external/ignored files from the current application source.
2. Decide and document the production late-delivery validation protocol; record the exact artifact provenance, package versions, feature list, threshold, and metrics.
3. Create a read-only characterization test suite around the current NLP pipeline, late model, preprocessing, label set, and threshold behavior. Fix the test dependency issue separately from migration.
4. Extract pure, import-safe Python modules for text cleaning, incident classification, late-delivery prediction, and (once supplied) supplier decisions. Preserve current model target and constraints.
5. Add Pydantic request/response schemas and FastAPI endpoints with explicit capability/error contracts. Load artifacts once at startup and validate artifact schemas.
6. Add API integration tests, model parity tests against known examples, input validation tests, and tests proving missing supplier-risk data is not invented.
7. Build the Next.js shell and capability-aware pages, then connect incident and delivery-risk pages to the API.
8. Add supplier pages only after the engine and data are integrated and constraint tests pass.
9. Add deployment configuration, pinned dependencies, health/readiness checks, observability, security controls, and artifact version management.
10. Compare old and new behavior on a fixed fixture set, obtain review approval, and only then deprecate any legacy UI/scripts.

## Audit conclusion

The current repository can support a focused migration of two model-serving capabilities: incident category classification and `Late_delivery_risk` prediction. It cannot currently support a faithful migration of a Streamlit dashboard, incident risk heuristic, supplier decision engine, risk-aware supplier recommendations, or general disruption prediction because those implementations are not present. Those missing components should be resolved before implementation is authorized.
