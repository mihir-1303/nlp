"""Order-time late-delivery prediction using the saved Person 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "person1_nlp"
    / "supply-chain-prediction"
    / "late_delivery_model.joblib"
)
DEFAULT_WARNING_THRESHOLD = 0.4

NUMERIC_FEATURES = (
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Latitude",
    "Longitude",
)
CATEGORICAL_FEATURES = (
    "Type",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Category Name",
    "Department Name",
    "Customer Segment",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class LateDeliveryModel(Protocol):
    """Prediction interface exposed by the saved scikit-learn pipeline."""

    classes_: Sequence[object]

    def predict_proba(self, features: pd.DataFrame) -> object: ...


@dataclass(frozen=True)
class LateDeliveryPrediction:
    """A model probability and its separately configured warning decision."""

    probability: float
    warning_threshold: float
    warning: bool


def load_late_delivery_model(
    model_path: Path = MODEL_PATH,
) -> LateDeliveryModel:
    """Load the existing model artifact from its repository-relative path."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Late-delivery model not found: {model_path}")

    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load late-delivery model at {model_path}: {exc}"
        ) from exc


def _validated_features(
    order_attributes: Mapping[str, object],
) -> pd.DataFrame:
    missing = [
        feature
        for feature in FEATURES
        if feature not in order_attributes
        or order_attributes[feature] is None
        or (
            isinstance(order_attributes[feature], str)
            and not order_attributes[feature].strip()
        )
    ]
    if missing:
        raise ValueError(
            "Required order attributes are missing or blank: "
            + ", ".join(missing)
        )

    values: dict[str, object] = {}
    for feature in NUMERIC_FEATURES:
        raw_value = order_attributes[feature]
        if isinstance(raw_value, bool):
            raise ValueError(f"{feature} must be a finite number.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be a valid number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{feature} must be finite.")
        values[feature] = value

    if values["Days for shipment (scheduled)"] < 0:
        raise ValueError("Days for shipment (scheduled) cannot be negative.")
    if values["Order Item Quantity"] <= 0:
        raise ValueError("Order Item Quantity must be greater than zero.")
    if values["Order Item Product Price"] < 0:
        raise ValueError("Order Item Product Price cannot be negative.")
    if values["Order Item Discount"] < 0:
        raise ValueError("Order Item Discount cannot be negative.")
    if not 0 <= values["Order Item Discount Rate"] <= 1:
        raise ValueError("Order Item Discount Rate must be between 0 and 1.")
    if not -90 <= values["Latitude"] <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= values["Longitude"] <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    for feature in CATEGORICAL_FEATURES:
        value = order_attributes[feature]
        if not isinstance(value, str):
            raise ValueError(f"{feature} must be text.")
        values[feature] = value.strip()

    return pd.DataFrame([{feature: values[feature] for feature in FEATURES}])


def predict_late_delivery(
    order_attributes: Mapping[str, object],
    *,
    warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
    model: LateDeliveryModel | None = None,
) -> LateDeliveryPrediction:
    """Validate order-time inputs and score them with the saved pipeline."""

    if isinstance(warning_threshold, bool):
        raise ValueError("Warning threshold must be a finite number from 0 to 1.")
    try:
        threshold = float(warning_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Warning threshold must be a finite number from 0 to 1."
        ) from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Warning threshold must be a finite number from 0 to 1.")

    features = _validated_features(order_attributes)
    predictor = model if model is not None else load_late_delivery_model()

    try:
        classes = list(predictor.classes_)
        if 0 not in classes or 1 not in classes:
            raise ValueError("The model must expose both target classes 0 and 1.")
        positive_class_index = classes.index(1)
        probabilities = np.asarray(predictor.predict_proba(features), dtype=float)
        if probabilities.shape != (1, len(classes)):
            raise ValueError("The model returned an unexpected probability shape.")
        probability = float(probabilities[0, positive_class_index])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("The model returned an invalid probability.")
    except Exception as exc:
        raise RuntimeError(f"Late-delivery prediction failed: {exc}") from exc

    return LateDeliveryPrediction(
        probability=probability,
        warning_threshold=threshold,
        warning=probability >= threshold,
    )
