from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_origins(value: str) -> tuple[str, ...]:
    origins = tuple(item.strip() for item in value.split(",") if item.strip())
    return origins or ("http://localhost:3000",)


@dataclass(frozen=True)
class Settings:
    nlp_model_path: Path = PROJECT_ROOT / "models" / "supply_chain_event_pipeline.joblib"
    delivery_model_path: Path = PROJECT_ROOT / "supply-chain-prediction" / "late_delivery_model.joblib"
    default_threshold: float = 0.4
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    @classmethod
    def from_environment(cls) -> "Settings":
        threshold_text = os.getenv("DELIVERY_THRESHOLD", "0.4")
        try:
            threshold = float(threshold_text)
        except ValueError as exc:
            raise ValueError("DELIVERY_THRESHOLD must be a number between 0 and 1") from exc
        if not 0 <= threshold <= 1:
            raise ValueError("DELIVERY_THRESHOLD must be a number between 0 and 1")
        return cls(
            nlp_model_path=Path(os.getenv("NLP_MODEL_PATH", str(PROJECT_ROOT / "models" / "supply_chain_event_pipeline.joblib"))),
            delivery_model_path=Path(os.getenv("DELIVERY_MODEL_PATH", str(PROJECT_ROOT / "supply-chain-prediction" / "late_delivery_model.joblib"))),
            default_threshold=threshold,
            cors_origins=_parse_origins(os.getenv("CORS_ORIGINS", "http://localhost:3000")),
        )
