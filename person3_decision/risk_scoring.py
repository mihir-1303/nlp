"""Transparent heuristic incident risk index; not a disruption probability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class RiskLevel(str, Enum):
    """Bands for the heuristic index, not calibrated operational risk levels."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"
    UNASSESSED = "Unassessed"


@dataclass(frozen=True)
class RiskScoringConfig:
    """Configurable provisional point mappings and inclusive band limits.

    The defaults are illustrative policy choices, not values learned from
    project outcomes or approved operational risk weights.
    """

    category_points: Mapping[str, int] = field(
        default_factory=lambda: {
            "Accident": 25,
            "Administrative Issue": 10,
            "Cyber Attack": 30,
            "Human Error": 15,
            "Others": 10,
            "Terrorism": 40,
            "Weather": 25,
            "Worker Strike": 30,
        }
    )
    severity_points: Mapping[str, int] = field(
        default_factory=lambda: {
            "Minor": 10,
            "Moderate": 25,
            "Severe": 45,
            "Extreme": 60,
        }
    )
    max_score: int = 100
    low_max: int = 24
    medium_max: int = 49
    high_max: int = 74

    def __post_init__(self) -> None:
        if isinstance(self.max_score, bool) or not isinstance(self.max_score, int):
            raise ValueError("max_score must be a positive integer.")
        if self.max_score < 1:
            raise ValueError("max_score must be a positive integer.")

        for field_name, points_by_label in (
            ("category_points", self.category_points),
            ("severity_points", self.severity_points),
        ):
            if not points_by_label:
                raise ValueError(f"{field_name} must not be empty.")
            for label, points in points_by_label.items():
                if not isinstance(label, str) or not label.strip():
                    raise ValueError(f"{field_name} labels must be non-empty strings.")
                if (
                    isinstance(points, bool)
                    or not isinstance(points, int)
                    or not 0 <= points <= self.max_score
                ):
                    raise ValueError(
                        f"{field_name}[{label!r}] must be an integer from "
                        f"0 to {self.max_score}."
                    )

        thresholds = (self.low_max, self.medium_max, self.high_max)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in thresholds):
            raise ValueError("Risk band limits must be integers.")
        if not (0 <= self.low_max < self.medium_max < self.high_max < self.max_score):
            raise ValueError(
                "Risk band limits must increase from zero and remain below max_score."
            )


@dataclass(frozen=True)
class RiskAssessment:
    """Heuristic index result; incomplete inputs are explicitly unassessed."""

    event_category: str | None
    severity: str | None
    score: int | None
    level: RiskLevel
    category_points: int | None
    severity_points: int | None
    notes: tuple[str, ...] = ()


DEFAULT_RISK_CONFIG = RiskScoringConfig()


def _normalized_label(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().casefold()


def _lookup_points(
    label: str | None,
    points_by_label: Mapping[str, int],
) -> int | None:
    normalized = _normalized_label(label)
    if normalized is None:
        return None
    return next(
        (
            points
            for configured_label, points in points_by_label.items()
            if configured_label.strip().casefold() == normalized
        ),
        None,
    )


def score_incident(
    event_category: str | None,
    severity: str | None,
    config: RiskScoringConfig = DEFAULT_RISK_CONFIG,
) -> RiskAssessment:
    """Score a known category/severity pair with configurable heuristic points.

    The complete-input score is the sum of category and severity points,
    capped at ``config.max_score``. Missing or unknown inputs produce no
    numeric score and an ``Unassessed`` level; they are never treated as zero.
    """

    category = event_category.strip() if isinstance(event_category, str) else None
    normalized_severity = severity.strip() if isinstance(severity, str) else None
    category_points = _lookup_points(category, config.category_points)
    severity_value = _lookup_points(normalized_severity, config.severity_points)
    notes: list[str] = []

    if category_points is None:
        notes.append(
            "Event category is missing or unrecognized; its contribution is unknown."
        )
    if severity_value is None:
        notes.append(
            "Severity is missing or unrecognized; its contribution is unknown."
        )

    if category_points is None or severity_value is None:
        return RiskAssessment(
            event_category=category,
            severity=normalized_severity,
            score=None,
            level=RiskLevel.UNASSESSED,
            category_points=category_points,
            severity_points=severity_value,
            notes=tuple(notes),
        )

    score = min(category_points + severity_value, config.max_score)
    if score <= config.low_max:
        level = RiskLevel.LOW
    elif score <= config.medium_max:
        level = RiskLevel.MEDIUM
    elif score <= config.high_max:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    return RiskAssessment(
        event_category=category,
        severity=normalized_severity,
        score=score,
        level=level,
        category_points=category_points,
        severity_points=severity_value,
        notes=(
            "Heuristic index only; not a calibrated probability or validated "
            "prediction of future disruption.",
        ),
    )
