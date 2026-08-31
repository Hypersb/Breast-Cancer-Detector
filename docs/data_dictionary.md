# Data Dictionary

## Dataset

This project uses the Wisconsin Breast Cancer dataset, which includes measurements describing cell nuclei from digitized breast mass images.

## Target variable

- `diagnosis`: binary target label
  - `B` = benign
  - `M` = malignant

## Feature groups

The dataset contains 30 numeric feature columns, grouped as follows:

### Mean features
- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `compactness_mean`
- `concavity_mean`
- `concave points_mean`
- `symmetry_mean`
- `fractal_dimension_mean`

### Standard error features
- `radius_se`
- `texture_se`
- `perimeter_se`
- `area_se`
- `smoothness_se`
- `compactness_se`
- `concavity_se`
- `concave points_se`
- `symmetry_se`
- `fractal_dimension_se`

### Worst features
- `radius_worst`
- `texture_worst`
- `perimeter_worst`
- `area_worst`
- `smoothness_worst`
- `compactness_worst`
- `concavity_worst`
- `concave points_worst`
- `symmetry_worst`
- `fractal_dimension_worst`

## Notes

- The dataset was cleaned before modeling by removing the non-feature metadata column and the `id` identifier column.
- The project treats `diagnosis` as the binary label to predict.
- This is a learning and portfolio project and should not be used as a clinical diagnosis system.
