import pandas as pd

from src.preprocessing import clean_dataset


def test_clean_dataset_removes_metadata_and_id_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "Unnamed: 32": [None, None],
            "diagnosis": ["B", "M"],
            "radius_mean": [10.0, 12.5],
            "texture_mean": [15.0, 18.0],
        }
    )

    cleaned = clean_dataset(df)

    assert "id" not in cleaned.columns
    assert "Unnamed: 32" not in cleaned.columns
    assert list(cleaned.columns) == ["diagnosis", "radius_mean", "texture_mean"]
    assert cleaned["diagnosis"].tolist() == ["B", "M"]


def test_clean_dataset_keeps_target_labels_trimmed():
    df = pd.DataFrame({
        "diagnosis": [" B ", "M", "  B  "]
    })

    cleaned = clean_dataset(df)

    assert cleaned["diagnosis"].tolist() == ["B", "M", "B"]
