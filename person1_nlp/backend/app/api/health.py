from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health(request: Request) -> dict[str, object]:
    statuses = request.app.state.model_loader.status()
    return {"status": "ok" if all(statuses.values()) else "degraded", "models": statuses}
