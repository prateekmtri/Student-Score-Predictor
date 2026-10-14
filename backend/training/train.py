"""Train and save the student score prediction pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    """Fit the prediction pipeline, report test metrics, and save the model."""
    training_dir = Path(__file__).resolve().parent
    data_path = training_dir / "data" / "processed" / "students_clean.csv"
    if not data_path.exists():
        raise SystemExit("Processed data not found. Run preprocess.py first.")

    data = pd.read_csv(data_path)
    features = data[["study_hours", "attendance"]]
    target = data["score"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"R2: {r2_score(y_test, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")

    regressor = model.named_steps["regressor"]
    for feature_name, coefficient in zip(features.columns, regressor.coef_):
        print(f"Coefficient ({feature_name}): {coefficient:.4f}")
    print(f"Intercept: {regressor.intercept_:.4f}")

    artifact_dir = training_dir.parent / "app" / "ml" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Model output path: {model_path}")


if __name__ == "__main__":
    main()
