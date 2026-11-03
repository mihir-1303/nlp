from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.delivery import router as delivery_router
from .api.health import router as health_router
from .api.incidents import router as incidents_router
from .core.config import Settings
from .core.model_loader import ModelLoader


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_environment()
    loader = ModelLoader(settings.nlp_model_path, settings.delivery_model_path)
    loader.load_all()
    app.state.settings = settings
    app.state.model_loader = loader
    app.state.logger = logging.getLogger("supply-chain-backend")
    yield


app = FastAPI(
    title="Supply Chain ML Backend",
    version="1.0.0",
    description="Serves existing incident-classification and Late_delivery_risk models.",
    lifespan=lifespan,
)
settings = Settings.from_environment()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(health_router)
app.include_router(incidents_router)
app.include_router(delivery_router)
