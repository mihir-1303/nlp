from pydantic import BaseModel, Field, field_validator


class IncidentClassificationRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=20_000)

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("incident description must not be empty")
        return value


class IncidentClassificationResponse(BaseModel):
    predicted_category: str
    model_confidence: float
    confidence_message: str
