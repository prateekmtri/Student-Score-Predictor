"""Generate a reproducible synthetic student score dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Generate, shuffle, and save clean and deliberately dirty student data."""
    rng = np.random.default_rng(42)
    study_hours = np.round(rng.uniform(1, 9, 80), 1)
    attendance = rng.integers(50, 101, 80)
    noise = rng.normal(0, 3, 80)
    scores = np.rint(
        np.clip(10 + 5 * study_hours + 0.4 * attendance + noise, 0, 100)
    ).astype(int)
    clean = pd.DataFrame(
        {"study_hours": study_hours, "attendance": attendance, "score": scores}
    )

    missing_attendance = clean.iloc[:3].copy()
    missing_attendance["attendance"] = np.nan

    out_of_range_attendance = clean.iloc[3:5].copy()
    out_of_range_attendance["attendance"] = [150, 120]

    negative_study_hours = clean.iloc[5:7].copy()
    negative_study_hours["study_hours"] = [-1.5, -2.0]

    duplicate_rows = clean.iloc[[7, 8, 9]].copy()
    data = pd.concat(
        [
            clean,
            missing_attendance,
            out_of_range_attendance,
            negative_study_hours,
            duplicate_rows,
        ],
        ignore_index=True,
    )
    data = data.iloc[rng.permutation(len(data))].reset_index(drop=True)

    output_dir = Path(__file__).resolve().parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "students.csv"
    data.to_csv(output_path, index=False, columns=["study_hours", "attendance", "score"])

    print(f"Total rows written: {len(data)}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
