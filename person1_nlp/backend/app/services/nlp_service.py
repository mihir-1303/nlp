from __future__ import annotations

import re
from typing import Any


def clean_incident_text(text: str) -> str:
    """Match the repository's GMRID text-cleaning transformation."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class NLPService:
    def __init__(self, model: Any) -> None:
        self.model = model

    def classify(self, description: str) -> tuple[str, float]:
        cleaned = clean_incident_text(description)
        if not cleaned:
            raise ValueError("incident description contains no usable text")
        prediction = self.model.predict([cleaned])[0]
        confidence = max(self.model.predict_proba([cleaned])[0])
        return str(prediction), float(confidence)
