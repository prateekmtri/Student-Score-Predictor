"""Load and provide access to the trained prediction model."""

import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)
model = None


def load_model():
    """Load the serialized model into the module-level model variable."""
    global model
    model_path = Path(__file__).resolve().parent / "artifacts" / "model.pkl"
    if not model_path.exists():
        logger.error("Model file not found at %s", model_path)
        raise FileNotFoundError("model.pkl not found. Run training/train.py first")
    model = joblib.load(model_path)
    return model


def get_model():
    """Return the loaded model or fail if loading has not occurred."""
    if model is None:
        raise RuntimeError("Model is not loaded")
    return model
