# Day 7 – Classification Models

## Objective
To implement and compare two classification algorithms using the same dataset and evaluation metrics.

## Dataset
**File:** `student_performance.csv`

- Total records: 24
- Input features: Age, Study_Hours, Attendance, Department
- Target variable: Result (Pass/Fail)
- Missing values were handled using suitable imputation techniques.

## Models Implemented
1. Logistic Regression
2. Decision Tree Classifier (maximum depth = 3)

## Data Splitting
- Training data: 80%
- Testing data: 20%
- Random state: 42
- Stratified sampling was used to maintain class proportions.

## Model Comparison

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Test Accuracy | 1.00 | 1.00 |
| Test Precision | 1.00 | 1.00 |
| Test Recall | 1.00 | 1.00 |
| Test F1-Score | 1.00 | 1.00 |
| Mean CV Training Accuracy | 1.00 | 1.00 |
| Mean CV Accuracy | 1.00 | 0.95 |
| Mean CV F1-Score | 1.00 | 0.96 |
| Performance Gap | 0.00 | 0.05 |

## Overfitting Analysis
- Logistic Regression showed no observed training-validation performance gap.
- Decision Tree showed a small performance gap of 0.05.
- The Decision Tree had an actual learned depth of 1 and 2 leaves.

## Selected Model: Logistic Regression

Logistic Regression was selected because it achieved higher cross-validation accuracy and F1-score than the Decision Tree.

It also showed no observed training-validation performance gap and is relatively simple to interpret.

## Conclusion
Both models achieved perfect scores on the test set. However, cross-validation results slightly favoured Logistic Regression.

The dataset contains only 24 records, so these results should be interpreted cautiously. More data would be needed to assess how well the models generalise.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook



