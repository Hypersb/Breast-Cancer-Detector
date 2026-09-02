# Breast Cancer Detection using Machine Learning

This repository contains a reproducible binary classification project for distinguishing benign and malignant breast cancer diagnoses from tabular diagnostic measurements. It is designed as a portfolio-ready machine learning project for education, experimentation, and internship applications.

## Project Overview

The workflow includes dataset validation, feature cleaning, exploratory data analysis, model benchmarking, model selection, and inference using a serialized artifact. The project emphasizes reproducibility, transparent documentation, and a clean engineering workflow.

## Problem Statement

Breast cancer diagnosis can be framed as a supervised binary classification problem using quantitative measurements from digitized breast mass images. This project trains a model to predict whether a tumor is likely benign or malignant based on the provided feature set.

## Dataset

This project uses the Wisconsin Diagnostic Breast Cancer (WDBC) dataset stored in `data/data.csv`.

### Verified characteristics

- Dataset name: Wisconsin Diagnostic Breast Cancer (WDBC)
- Creator/source: William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian, along with the broader UCI Machine Learning Repository dataset curation effort
- Source repository: UCI Machine Learning Repository (widely distributed in scikit-learn and public ML benchmarks)
- Observations: 569
- Feature count: 30 numeric predictors
- Target variable: `diagnosis`
- Target classes: `B` = benign, `M` = malignant
- Data quality: no missing values in the shipped CSV, and the dataset includes a non-feature identifier column (`id`) that is removed during cleaning

The dataset in this repository is consistent with the standard WDBC tabular dataset used in many public machine learning examples and educational materials.

### Citation and usage note

A standard citation for the dataset is:

> Wolberg, W.H., Street, W.N., and Mangasarian, O.L. (1995). Breast cancer Wisconsin (diagnostic) data set. UCI Machine Learning Repository.

The source code in this repository is MIT-licensed, but the dataset itself is not re-licensed by this project. Please review the original repository or dataset source for dataset-specific usage terms before reusing it in another context.

## ML Pipeline

1. Load and validate the raw CSV
2. Remove non-feature metadata columns such as `id` and empty trailing columns
3. Standardize/prepare feature matrices for modeling
4. Split data into training and test sets with stratification
5. Train several baseline classifiers
6. Evaluate metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC
7. Save the best-scoring model artifact to `models/best_model.pkl`
8. Use the model for inference via `src/predict.py`

## Repository Structure

```text
breast-cancer-detector/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── data.csv
├── docs/
│   └── data_dictionary.md
├── models/
│   └── best_model.pkl
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── reports/
│   ├── correlation_heatmap.png
│   ├── diagnosis_distribution.png
│   └── top_correlations.png
├── src/
│   ├── eda.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── tune.py
├── tests/
│   ├── test_predict.py
│   └── test_preprocessing.py
├── .editorconfig
├── .gitignore
├── LICENSE
├── Makefile
├── MODEL_CARD.md
├── README.md
├── pyproject.toml
├── requirements.txt
└── train_clean.txt
```

## Models Evaluated

The project compares multiple classifiers and records validation metrics on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9823 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| Random Forest | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9942 |
| Gradient Boosting | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9947 |

## Evaluation Metrics

The project uses standard binary classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Results

The best verified model in this repository is a logistic regression pipeline trained with a standard scaler.

- Accuracy: 0.9649
- Precision: 0.9750
- Recall: 0.9286
- F1-score: 0.9512
- ROC-AUC: 0.9960

The trained model artifact is saved at `models/best_model.pkl`.

> Compatibility note: the serialized model was created with scikit-learn 1.5.2. The project dependencies are pinned to this version to reduce compatibility issues with the saved artifact.

## Installation

### Option 1: Create a virtual environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: Use the included Makefile

```bash
make install
```

## Usage

Run the project modules from the repository root:

```bash
python src/preprocessing.py
python src/eda.py
python src/train.py
python src/tune.py
python src/predict.py
```

You can also use the Make targets:

```bash
make train
make eda
make predict
```

## Running Predictions

```bash
python src/predict.py
```

The prediction script loads the saved model and returns a probabilistic classification for a sample record built from the dataset features.

## Running Tests

```bash
pytest -v
```

The project includes automated checks for preprocessing and prediction behavior.

## Reproducibility

This project is designed to be reproducible with a locked dependency set. The key compatibility requirement is:

- scikit-learn == 1.5.2
- Python >= 3.10 and < 3.14

This keeps the saved model artifact and training pipeline consistent with one another.

## Limitations

- The project is intended for educational and portfolio use.
- It is not a medical diagnostic system.
- The model is trained on a benchmark dataset and may not generalize to all populations or clinical settings.
- Predictions should not replace professional medical review or clinical decision-making.

## Ethical / Medical Disclaimer

This project is for educational and research purposes only and is not a medical diagnostic system. Predictions should not be used for clinical decision-making.

## Future Improvements

- Add feature importance and model explainability analysis
- Add experiment tracking and model versioning
- Extend the project with a simple API or CLI
- Improve notebook-based analysis and reporting
- Add more rigorous validation for deployment-oriented workflows

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The dataset used in this project remains subject to its respective original licensing and usage terms. The project source code and the dataset are distinct intellectual property domains and should be handled separately.
