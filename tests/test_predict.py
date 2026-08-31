import pandas as pd

from src.predict import get_feature_columns, prepare_features, predict_from_record


def test_get_feature_columns_returns_expected_count():
    columns = get_feature_columns()

    assert len(columns) == 30
    assert "diagnosis" not in columns
    assert columns[0] == "radius_mean"


def test_prepare_features_uses_expected_feature_order():
    feature_columns = get_feature_columns()
    sample = {column: float(idx) for idx, column in enumerate(feature_columns)}

    prepared = prepare_features(sample)

    assert list(prepared.columns) == feature_columns
    assert prepared.shape == (1, 30)


def test_predict_from_record_returns_prediction_dict():
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

    assert set(result.keys()) == {"prediction", "probability_malignant", "probability_benign"}
    assert result["prediction"] in {"Benign", "Malignant"}
    assert 0.0 <= result["probability_benign"] <= 1.0
    assert 0.0 <= result["probability_malignant"] <= 1.0
