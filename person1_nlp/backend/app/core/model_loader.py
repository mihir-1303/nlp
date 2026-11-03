from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib


DELIVERY_FEATURES = (
    "Days for shipment (scheduled)", "Order Item Quantity", "Order Item Product Price",
    "Order Item Discount", "Order Item Discount Rate", "Latitude", "Longitude",
    "Type", "Shipping Mode", "Market", "Order Region", "Category Name",
    "Department Name", "Customer Segment",
)


class ModelLoadError(RuntimeError):
    """Internal error raised when a model cannot be loaded or validated."""


@dataclass(frozen=True)
class ModelStatus:
    loaded: bool
    error: str | None = None


class ModelLoader:
    """Loads and validates both trusted joblib artifacts once per app process."""

    def __init__(self, nlp_path: Path, delivery_path: Path) -> None:
        self._nlp_path = nlp_path
        self._delivery_path = delivery_path
        self.nlp_model: Any | None = None
        self.delivery_model: Any | None = None
        self.nlp_status = ModelStatus(False)
        self.delivery_status = ModelStatus(False)

    def load_all(self) -> None:
        self.nlp_model, self.nlp_status = self._load_one(self._nlp_path, "NLP", self._validate_nlp)
        self.delivery_model, self.delivery_status = self._load_one(self._delivery_path, "delivery", self._validate_delivery)

    def _load_one(self, path: Path, name: str, validator: Callable[[Any], None]) -> tuple[Any | None, ModelStatus]:
        try:
            if not path.is_file():
                raise ModelLoadError(f"{name} model file is missing")
            model = joblib.load(path)
            validator(model)
            return model, ModelStatus(True)
        except Exception as exc:
            return None, ModelStatus(False, str(exc))

    @staticmethod
    def _validate_nlp(model: Any) -> None:
        if not callable(getattr(model, "predict", None)):
            raise ModelLoadError("NLP model does not support prediction")
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelLoadError("NLP model does not support confidence probabilities")

    @staticmethod
    def _validate_delivery(model: Any) -> None:
        if not callable(getattr(model, "predict", None)):
            raise ModelLoadError("delivery model does not support prediction")
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelLoadError("delivery model does not support probabilities")
        if tuple(getattr(model, "feature_names_in_", ())) != DELIVERY_FEATURES:
            raise ModelLoadError("delivery model feature schema does not match the application")

    def status(self) -> dict[str, bool]:
        return {"nlp_model_loaded": self.nlp_status.loaded, "delivery_model_loaded": self.delivery_status.loaded}
