# Breast Cancer Detection

This repository contains a machine learning project for classifying breast tumors as malignant or benign using the Wisconsin Breast Cancer dataset.

## Project status

This project is in active development.

## Project structure

- `data/` contains the dataset
- `notebooks/` contains exploratory and modeling notebooks
- `src/` contains reusable Python modules
- `models/` stores trained model artifacts

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data source

The project uses the breast cancer dataset already available in the workspace.

## Goal

Build a reproducible classification pipeline that predicts whether a tumor is malignant or benign, with attention to model evaluation and responsible interpretation.
