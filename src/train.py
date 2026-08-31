from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from src.preprocessing import clean_dataset, load_dataset
except ModuleNotFoundError:  # pragma: no cover
    from preprocessing import clean_dataset, load_dataset


def get_train_test_data(data_path: str | Path):
    df = clean_dataset(load_dataset(data_path))
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"].map({"B": 0, "M": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test


def build_models() -> dict[str, Pipeline]:
    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }
    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "data.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = get_train_test_data(data_path)
    model_dict = build_models()
    results = []

    for model_name, model in model_dict.items():
        model.fit(X_train, y_train)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
        metrics = evaluate_model(model, X_test, y_test)
        metrics["cv_roc_auc_mean"] = cv_score
        metrics["model_name"] = model_name
        results.append(metrics)

        print(f"\n=== {model_name.upper()} ===")
        print(f"CV ROC-AUC mean: {cv_score:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print("Confusion matrix:\n", metrics["confusion_matrix"])

    best_result = max(results, key=lambda r: (r["roc_auc"], r["f1"], r["accuracy"]))
    best_model_name = best_result["model_name"]
    best_model = model_dict[best_model_name]
    joblib.dump(best_model, models_dir / "best_model.pkl")

    print(f"\nBEST MODEL: {best_model_name}")
    print(f"Best ROC-AUC: {best_result['roc_auc']:.4f}")
    print(f"Best F1: {best_result['f1']:.4f}")


if __name__ == "__main__":
    main()
