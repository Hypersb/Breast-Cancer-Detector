from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the breast cancer dataset from CSV."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the initial data cleaning steps for this project."""
    cleaned = df.copy()

    # Remove the empty trailing column produced by the CSV export.
    cleaned = cleaned.drop(columns=[col for col in cleaned.columns if "Unnamed:" in str(col)], errors="ignore")

    # The dataset contains an identifier that should not be used as a feature.
    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])

    # Normalize the target labels and remove any accidental whitespace.
    if "diagnosis" in cleaned.columns:
        cleaned["diagnosis"] = cleaned["diagnosis"].astype(str).str.strip()

    return cleaned


if __name__ == "__main__":
    df = load_dataset(Path(__file__).resolve().parents[1] / "data" / "data.csv")
    cleaned = clean_dataset(df)
    print(f"shape={cleaned.shape}")
    print(f"target_counts={cleaned['diagnosis'].value_counts().to_dict()}")
    print(cleaned.head())
