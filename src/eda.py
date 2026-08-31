from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from src.preprocessing import clean_dataset, load_dataset
except ModuleNotFoundError:  # pragma: no cover
    from preprocessing import clean_dataset, load_dataset


def summarize_target(df: pd.DataFrame) -> pd.Series:
    return df["diagnosis"].value_counts(normalize=True).sort_index()


def plot_diagnosis_distribution(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    value_counts = df["diagnosis"].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(data=pd.DataFrame({"diagnosis": value_counts.index, "count": value_counts.values}), x="diagnosis", y="count", hue="diagnosis", palette=["#45aaf2", "#ff6b6b"], dodge=False, legend=False)
    plt.title("Diagnosis distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show(block=False)


def plot_top_correlations(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    X = df.drop(columns=["diagnosis"])
    corr = X.corrwith(df["diagnosis"].map({"B": 0, "M": 1})).sort_values(ascending=False)
    top = corr.head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame({"feature": top.index, "corr": top.values}), x="corr", y="feature", hue="feature", palette="viridis", dodge=False, legend=False)
    plt.title("Top 10 feature correlations with diagnosis")
    plt.xlabel("Correlation with diagnosis")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show(block=False)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | Path | None = None) -> None:
    target_map = {"B": 0, "M": 1}
    encoded = df.copy()
    encoded["diagnosis"] = encoded["diagnosis"].map(target_map)
    corr = encoded.corr(numeric_only=True)
    selected = corr["diagnosis"].abs().sort_values(ascending=False).head(10).index
    subset = encoded[selected].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(subset, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation heatmap for top diagnostic features")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show(block=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    data_path = project_root / "data" / "data.csv"
    df = clean_dataset(load_dataset(data_path))
    print("Dataset shape:", df.shape)
    print("Target distribution:\n", summarize_target(df))
    print("Missing values:\n", df.isna().sum().sum())
    print("Duplicates:", int(df.duplicated().sum()))

    plot_diagnosis_distribution(df, save_path=reports_dir / "diagnosis_distribution.png")
    plot_top_correlations(df, save_path=reports_dir / "top_correlations.png")
    plot_correlation_heatmap(df, save_path=reports_dir / "correlation_heatmap.png")


if __name__ == "__main__":
    main()
