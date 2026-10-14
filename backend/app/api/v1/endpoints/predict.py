"""Prediction API endpoint."""

from fastapi import APIRouter, HTTPException

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.predict_service import predict_score

router = APIRouter()


@router.post("/", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Return a predicted score for the supplied student features."""
    try:
        value = predict_score(request.study_hours, request.attendance)
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Model is not available. Run training first.",
        ) from None
    return {"predicted_score": value}
