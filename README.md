# Breast Cancer Detection

A portfolio-style machine learning project for classifying breast tumors as malignant or benign using the Wisconsin Breast Cancer dataset.

This project is educational in nature and is not a medical diagnostic tool.

## Problem statement

The goal is to build a reproducible binary classification workflow that predicts whether a tumor is malignant or benign using tabular medical features from the dataset.

## Dataset overview

The dataset is already present in the workspace at `data/data.csv`.

### Verified dataset findings

Using the dataset as it exists in the workspace, we verified:

- Shape: `(569, 31)` after removing the empty trailing CSV column and the identifier column
- Target column: `diagnosis`
- Target classes: `B` and `M`
- Class distribution:
  - `B`: 357
  - `M`: 212
- Missing values: `0`
- Duplicate rows: `0`

The dataset is moderately imbalanced, with benign cases making up about 62.7% and malignant cases about 37.3%.

## Project structure

```text
breast cancer detector/
├── data/
│   └── data.csv
├── notebooks/
│   └── 01_data_understanding.ipynb
├── src/
│   ├── eda.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── tune.py
├── models/
│   └── best_model.pkl
├── reports/
│   ├── correlation_heatmap.png
│   ├── diagnosis_distribution.png
│   └── top_correlations.png
├── .gitignore
├── README.md
├── requirements.txt
└── .venv/
```

## Environment setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the workflow

```bash
# dataset inspection / cleaning validation
python src/preprocessing.py

# exploratory data analysis
python src/eda.py

# baseline model comparison
python src/train.py

# tuned model search
python src/tune.py

# example prediction
python src/predict.py
```

## Methodology

1. Load and inspect the dataset
2. Remove the empty trailing CSV column and ID field
3. Check target distribution and class balance
4. Perform EDA and visualize the key relationships
5. Split the data into train and test sets with stratification
6. Train several classification models
7. Compare models using accuracy, precision, recall, F1, confusion matrix, and ROC-AUC
8. Tune promising models with focused GridSearchCV
9. Save the best-performing pipeline
10. Use the saved model for inference

## Baseline model results

These results were produced from verified runs on the actual dataset in this workspace.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9823 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| Random Forest | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9942 |
| Gradient Boosting | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9947 |

The best baseline model on the verified runs was logistic regression based on ROC-AUC and F1 performance.

## Model tuning

The project includes a tuning workflow in `src/tune.py` using `GridSearchCV` for the strongest candidates.

The tuning step is intentionally focused and uses a small search space to avoid unnecessary complexity while keeping the validation process reproducible.

## Prediction pipeline

The saved best model can be used via `src/predict.py`.

This script loads the trained model and makes predictions from a feature dictionary representing a single tumor observation.

## Important note

This project is designed as a machine-learning and portfolio exercise. It is not a medical diagnostic system and should not be used to make real clinical decisions.

## Technologies used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- Jupyter
- Git / GitHub

## Future improvements

- Add more notebooks for preprocessing, feature engineering, and model interpretation
- Add cross-validation comparison across more models
- Generate feature-importance plots for the final model
- Add a small command-line interface for batch prediction
- Improve documentation and experiment tracking
