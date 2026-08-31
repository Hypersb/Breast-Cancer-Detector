from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.preprocessing import clean_dataset, load_dataset
except ModuleNotFoundError:  # pragma: no cover
    from preprocessing import clean_dataset, load_dataset


def get_split(data_path: str | Path):
    df = clean_dataset(load_dataset(data_path))
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].map({"B": 0, "M": 1})

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )


def build_search_spaces():
    search_spaces = {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["liblinear", "lbfgs"],
            "model__max_iter": [200, 500, 2000],
        },
        "svm": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf"],
        },
    }
    return search_spaces


def build_models():
    return {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))]
        ),
        "svm": Pipeline(
            [("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=42))]
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "data.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = get_split(data_path)
    model_map = build_models()
    spaces = build_search_spaces()

    best_model_name = None
    best_model = None
    best_score = -1.0
    best_params = None

    for name, model in model_map.items():
        search = GridSearchCV(
            estimator=model,
            param_grid=spaces[name],
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        print(f"\n=== {name.upper()} TUNING ===")
        print("Best params:", search.best_params_)
        print("Best CV ROC-AUC:", round(search.best_score_, 4))

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_model_name = name
            best_params = search.best_params_

    if best_model is not None:
        final_score = best_model.score(X_test, y_test)
        print(f"\nBEST MODEL AFTER TUNING: {best_model_name}")
        print("Best params:", best_params)
        print("Test accuracy:", round(final_score, 4))
        joblib.dump(best_model, models_dir / "best_model.pkl")


if __name__ == "__main__":
    main()
