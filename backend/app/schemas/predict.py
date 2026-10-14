"""Request and response schemas for score predictions."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input features for a student score prediction."""

    study_hours: float = Field(..., ge=0, le=24)
    attendance: float = Field(..., ge=0, le=100)


class PredictResponse(BaseModel):
    """Predicted student score response."""

    predicted_score: float
