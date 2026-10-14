"""Clean the synthetic student score dataset."""

from pathlib import Path

import pandas as pd


def main():
    """Apply ordered data quality filters and save the cleaned dataset."""
    training_dir = Path(__file__).resolve().parent
    raw_path = training_dir / "data" / "raw" / "students.csv"
    if not raw_path.exists():
        raise SystemExit("Raw data not found. Run generate_data.py first.")

    data = pd.read_csv(raw_path)
    print(f"Starting row count: {len(data)}")

    previous_count = len(data)
    data = data.drop_duplicates()
    print(f"Exact duplicates removed: {previous_count - len(data)}")

    previous_count = len(data)
    data = data.dropna()
    print(f"Rows with missing values removed: {previous_count - len(data)}")

    previous_count = len(data)
    data = data[data["attendance"].between(0, 100)]
    print(f"Rows with invalid attendance removed: {previous_count - len(data)}")

    previous_count = len(data)
    data = data[data["study_hours"].between(0, 24)]
    print(f"Rows with invalid study_hours removed: {previous_count - len(data)}")

    previous_count = len(data)
    data = data[data["score"].between(0, 100)]
    print(f"Rows with invalid score removed: {previous_count - len(data)}")
    print(f"Final row count: {len(data)}")

    output_dir = training_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "students_clean.csv"
    data.to_csv(
        output_path,
        index=False,
        columns=["study_hours", "attendance", "score"],
    )
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
