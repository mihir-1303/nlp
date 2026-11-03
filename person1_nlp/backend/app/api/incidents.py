from fastapi import APIRouter, HTTPException, Request, status

from ..schemas.incident import IncidentClassificationRequest, IncidentClassificationResponse
from ..services.nlp_service import NLPService

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


@router.post("/classify", response_model=IncidentClassificationResponse)
def classify_incident(payload: IncidentClassificationRequest, request: Request) -> IncidentClassificationResponse:
    model = request.app.state.model_loader.nlp_model
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="incident classification model is unavailable")
    try:
        category, confidence = NLPService(model).classify(payload.description)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        request.app.state.logger.exception("Incident classification failed")
        raise HTTPException(status_code=500, detail="incident classification failed") from exc
    return IncidentClassificationResponse(
        predicted_category=category,
        model_confidence=confidence,
        confidence_message="Model confidence is confidence in the predicted incident category; it is not disruption probability.",
    )
