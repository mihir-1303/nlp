from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.model_loader import DELIVERY_FEATURES
from ..schemas.delivery import DeliveryPredictionRequest


class DeliveryService:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, request: DeliveryPredictionRequest, threshold: float) -> tuple[float, int, bool]:
        values = request.model_dump(by_alias=True, exclude={"threshold"})
        frame = pd.DataFrame([values], columns=DELIVERY_FEATURES)
        probability = float(self.model.predict_proba(frame)[0][1])
        prediction = int(probability >= threshold)
        return probability, prediction, bool(prediction)
