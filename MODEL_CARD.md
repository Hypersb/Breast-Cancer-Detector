# Model Card: Breast Cancer Detector

## Model summary

This project trains a binary classifier to detect whether a tumor is malignant or benign using numerical measurements derived from breast cell nuclei.

## Intended use

- Educational and portfolio use
- Binary classification demonstration on tabular medical data
- Research and experimentation

## Out-of-scope use

- Clinical triage or diagnosis
- Real-world patient decision-making
- High-stakes medical use without expert review

## Data

The model is trained on the Wisconsin Breast Cancer dataset, which includes 30 morphological features and a binary diagnosis label.

## Performance

The verified best model was a logistic regression pipeline trained and evaluated in this repo.

- Accuracy: 0.9649
- Precision: 0.9750
- Recall: 0.9286
- F1-score: 0.9512
- ROC-AUC: 0.9960

## Limitations

- The dataset is moderately imbalanced, with more benign than malignant samples.
- The model is trained on a tabular dataset from a single benchmark source and may not generalize to all populations.
- This project is not a replacement for medical diagnosis.

## Ethical considerations

This model should be used only for education, experimentation, and exploratory learning. It must not be used as the sole basis for medical decisions.
