from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from src.preprocessing import clean_dataset, load_dataset
except ModuleNotFoundError:  # pragma: no cover
    from preprocessing import clean_dataset, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"


def get_feature_columns() -> list[str]:
    df = clean_dataset(load_dataset(DATA_PATH))
    return [col for col in df.columns if col != "diagnosis"]


def prepare_features(raw: dict[str, float] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        df = pd.DataFrame([raw])

    required = get_feature_columns()
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return df[required]


def predict_from_record(raw: dict[str, float]) -> dict[str, object]:
    model = joblib.load(MODEL_PATH)
    X = prepare_features(raw)
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    return {
        "prediction": "Malignant" if prediction == 1 else "Benign",
        "probability_malignant": float(probabilities[1]),
        "probability_benign": float(probabilities[0]),
    }


def main() -> None:
    example = {
        "radius_mean": 14.0,
        "texture_mean": 20.0,
        "perimeter_mean": 90.0,
        "area_mean": 600.0,
        "smoothness_mean": 0.10,
        "compactness_mean": 0.10,
        "concavity_mean": 0.08,
        "concave points_mean": 0.04,
        "symmetry_mean": 0.20,
        "fractal_dimension_mean": 0.06,
        "radius_se": 0.5,
        "texture_se": 1.0,
        "perimeter_se": 3.0,
        "area_se": 30.0,
        "smoothness_se": 0.01,
        "compactness_se": 0.03,
        "concavity_se": 0.03,
        "concave points_se": 0.01,
        "symmetry_se": 0.02,
        "fractal_dimension_se": 0.006,
        "radius_worst": 16.0,
        "texture_worst": 24.0,
        "perimeter_worst": 110.0,
        "area_worst": 800.0,
        "smoothness_worst": 0.12,
        "compactness_worst": 0.18,
        "concavity_worst": 0.15,
        "concave points_worst": 0.08,
        "symmetry_worst": 0.30,
        "fractal_dimension_worst": 0.09,
    }

    result = predict_from_record(example)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
