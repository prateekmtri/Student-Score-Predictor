"""Prediction service for the student score model."""

import pandas as pd

from app.ml.model_loader import get_model


def predict_score(study_hours: float, attendance: float) -> float:
    """Predict and bound a score using the loaded pipeline."""
    features = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["study_hours", "attendance"],
    )
    prediction = float(get_model().predict(features)[0])
    return round(min(max(prediction, 0.0), 100.0), 1)
