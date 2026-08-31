# Breast Cancer Detector

A reproducible machine learning project for binary tumor classification using the Wisconsin Breast Cancer dataset.

This repository demonstrates a complete end-to-end ML workflow: data cleaning, exploratory analysis, model selection, evaluation, artifact saving, and inference. It is intended for learning, experimentation, and portfolio use, not for clinical decision-making.

## Project goals

- Build a dependable binary classification pipeline for malignant vs. benign tumor prediction
- Use a real tabular dataset and evaluate models with clinically meaningful metrics
- Demonstrate a clean engineering workflow with tests, documentation, and CI
- Present a portfolio-ready ML project with reproducible setup and transparent limitations

## Problem statement

Breast cancer screening and classification require identifying patterns in diagnostic measurements from tumor cell nuclei. This project models that problem as a supervised binary classification task using structured numerical features.

## Dataset

The project uses the Wisconsin Breast Cancer dataset stored in `data/data.csv`.

### Dataset characteristics

- Rows: 569
- Features: 30 numeric predictive features
- Target: `diagnosis`
- Classes: `B` (benign), `M` (malignant)
- Missing values: 0
- Duplicate rows: 0

The dataset is mildly imbalanced, with 357 benign samples and 212 malignant samples.

## Repository structure

```text
breast cancer detector/
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
├── Makefile
├── MODEL_CARD.md
├── README.md
├── pyproject.toml
├── requirements.txt
└── .venv/
```

## Setup

### Option 1: Standard Python environment

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Makefile workflow

```bash
make install
```

## Run the project

```bash
python src/preprocessing.py
python src/eda.py
python src/train.py
python src/tune.py
python src/predict.py
```

or

```bash
make train
make eda
make predict
```

## Methodology

1. Load the raw dataset and validate the schema
2. Drop non-feature metadata columns such as the identifier and empty trailing columns
3. Inspect class balance and target distribution
4. Conduct exploratory data analysis with visual summaries
5. Split data into train and test sets using stratification
6. Train several classification models
7. Evaluate on accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC
8. Save the best-performing model artifact
9. Reuse the model for inference through a prediction pipeline

## Model evaluation

The project evaluates multiple baseline classifiers. Verified results from this repo are shown below.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9823 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 |
| Random Forest | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9942 |
| Gradient Boosting | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9947 |

### Best model

The best verified model is the logistic regression pipeline.

- Accuracy: 0.9649
- Precision: 0.9750
- Recall: 0.9286
- F1-score: 0.9512
- ROC-AUC: 0.9960

The model artifact is saved in `models/best_model.pkl`.

## Testing and validation

This repository includes automated tests for preprocessing and prediction flows.

```bash
pytest -q
```

CI is configured in `.github/workflows/ci.yml` to run on pushes and pull requests.

## Model card and documentation

For project-level context, see:

- [MODEL_CARD.md](MODEL_CARD.md)
- [docs/data_dictionary.md](docs/data_dictionary.md)

## Key considerations

- This project is a learning and portfolio ML application.
- It should not be used as a clinical diagnostic system.
- The model is trained on a benchmark dataset and may not generalize beyond that distribution.

## Technologies

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- joblib
- Jupyter
- GitHub Actions
- pytest

## Future improvements

- Add feature-importance and explainability analysis
- Add model version tracking and experiment logging
- Create a lightweight CLI or REST API for inference
- Expand notebook coverage for feature engineering and validation
- Add more production-style monitoring and deployment scaffolding

## License

This project is intended for educational and portfolio use. Add a formal license if you plan to distribute it beyond personal or portfolio contexts.
