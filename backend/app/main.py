"""FastAPI application for the Student Score Predictor."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import predict
from app.ml.model_loader import load_model

logger = logging.getLogger(__name__)

app = FastAPI(title="Student Score Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Load the model on startup while allowing the server to remain available."""
    try:
        load_model()
    except Exception as exc:
        logger.warning("Model could not be loaded; prediction endpoint unavailable: %s", exc)


app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predict"])


@app.get("/")
def health_check():
    """Return a simple application health status."""
    return {"status": "ok"}
