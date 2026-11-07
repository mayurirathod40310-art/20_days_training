# Day 4: Data Leakage Checklist

## 1. Dataset Audit
- [x] Checked duplicate rows.
- [x] Checked duplicate Student IDs.
- [x] Checked missing values.
- [x] Examined target class distribution.

## 2. Train-Test Leakage Checks
- [x] Used separate training and testing datasets.
- [x] Checked for common Student IDs.
- [x] Used stratified splitting.
- [x] Used a fixed random_state.

## 3. Common Causes of Data Leakage

1. Using test data during model training.
2. Performing preprocessing on the complete dataset before splitting.
3. Including target-related information in input features.
4. Using future information to predict past outcomes.
5. Allowing duplicate records across training and testing datasets.

## 4. How Leakage Can Inflate Results

Data leakage allows a machine learning model to access information that would not normally be available during real-world prediction.

This can lead to artificially high accuracy and unreliable evaluation results.

## 5. Prevention Methods

- Split data before preprocessing.
- Fit preprocessing steps only on training data.
- Use pipelines for preprocessing and model training.
- Remove or investigate duplicate records.
- Keep the test dataset separate until final evaluation.

## 6. Conclusion

A proper data audit and splitting strategy help prevent data leakage and ensure reliable machine learning evaluation.

