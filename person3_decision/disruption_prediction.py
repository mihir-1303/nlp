"""Operational disruption prediction availability contract.

The current project datasets contain incident/event labels, but no linked
order-, shipment-, or supplier-level disruption outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DisruptionPredictionStatus(str, Enum):
    """Availability of an operational disruption prediction."""

    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DisruptionPrediction:
    """Explicitly unavailable operational prediction and its prerequisites."""

    status: DisruptionPredictionStatus
    reason: str
    required_data: tuple[str, ...]

    @property
    def probability(self) -> None:
        """No probability is available under the current data contract."""

        return None


DISRUPTION_PREDICTION_UNAVAILABLE = DisruptionPrediction(
    status=DisruptionPredictionStatus.UNAVAILABLE,
    reason=(
        "The repository has no verified order-, shipment-, or supplier-linked "
        "disruption outcome target, so an operational disruption prediction "
        "cannot be trained or evaluated."
    ),
    required_data=(
        "Stable order, shipment, and supplier identifiers with a verified "
        "incident-to-operational-record linkage.",
        "A prediction timestamp and a domain-defined future prediction horizon.",
        "Auditable disruption outcome labels for that unit and horizon, "
        "including non-disrupted examples and outcome timestamps.",
        "For impact prediction, observed delay duration or financial loss with "
        "defined units, currency, and measurement window.",
    ),
)
