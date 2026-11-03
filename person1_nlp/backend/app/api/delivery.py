from fastapi import APIRouter, HTTPException, Request, status

from ..schemas.delivery import DeliveryPredictionRequest, DeliveryPredictionResponse
from ..services.delivery_service import DeliveryService

router = APIRouter(prefix="/api/delivery", tags=["delivery"])


@router.post("/predict", response_model=DeliveryPredictionResponse)
def predict_delivery(payload: DeliveryPredictionRequest, request: Request) -> DeliveryPredictionResponse:
    model = request.app.state.model_loader.delivery_model
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="late-delivery model is unavailable")
    try:
        threshold = payload.threshold
        if threshold is None:
            threshold = request.app.state.settings.default_threshold
        probability, prediction, warning = DeliveryService(model).predict(payload, threshold)
    except Exception as exc:
        request.app.state.logger.exception("Delivery prediction failed")
        raise HTTPException(status_code=500, detail="delivery prediction failed") from exc
    return DeliveryPredictionResponse(
        estimated_late_delivery_probability=probability,
        prediction=prediction,
        warning=warning,
        threshold_used=threshold,
    )
