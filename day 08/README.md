# Day 8 – Metrics and Imbalanced Data

## 1. Objective

The objective of this practical is to understand and apply different classification evaluation metrics, evaluate model performance on an imbalanced dataset, and understand how class weighting and threshold selection affect classification results.

## 2. Dataset Description

A synthetic binary classification dataset was generated using Scikit-learn's `make_classification()` function.

- Total records: 1,000
- Total features: 5
- Target classes: 0 and 1
- Class 0: 894 records (89.4%)
- Class 1: 106 records (10.6%)

The dataset is imbalanced because Class 0 has significantly more records than Class 1.

The data was divided into training and testing sets using an 80:20 split with stratification.

- Training records: 800
- Testing records: 200

## 3. Evaluation Metrics

The following metrics were used to evaluate the classification models:

- **Confusion Matrix:** Shows the number of correct and incorrect predictions using True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).
- **Accuracy:** Measures the proportion of total predictions that are correct.
- **Precision:** Measures how many predicted positive cases are actually positive.
- **Recall:** Measures how many actual positive cases are correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across different thresholds.
- **PR-AUC (Average Precision):** Summarizes the precision-recall performance, especially useful for imbalanced datasets.

## 4. Baseline Logistic Regression Results

A Logistic Regression model was trained without class weighting.

| Metric | Score |
|---|---:|
| Accuracy | 0.975 |
| Precision | 1.000 |
| Recall | 0.762 |
| F1-Score | 0.865 |
| ROC-AUC | 0.950 |
| PR-AUC | 0.920 |

The baseline model achieved high accuracy and precision. However, its recall was 0.762, meaning it did not identify all actual positive cases.

The confusion matrix was:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 179 (TN) | 0 (FP) |
| Actual 1 | 5 (FN) | 16 (TP) |

## 5. Class Weighting

A second Logistic Regression model was trained using:

`class_weight="balanced"`

This gives greater importance to the minority class during model training.

### Model Comparison

| Metric | Baseline | Class-Weighted |
|---|---:|---:|
| Accuracy | 0.975 | 0.950 |
| Precision | 1.000 | 0.704 |
| Recall | 0.762 | 0.905 |
| F1-Score | 0.865 | 0.792 |
| ROC-AUC | 0.950 | 0.956 |
| PR-AUC | 0.920 | 0.921 |

### Observation

- The class-weighted model improved recall from 0.762 to 0.905.
- Precision decreased from 1.000 to 0.704.
- Accuracy decreased slightly from 0.975 to 0.950.
- ROC-AUC and PR-AUC improved slightly.

Class weighting helped the model identify more minority-class cases, but it also increased false-positive predictions.

## 6. Threshold Selection

The baseline model's predicted probabilities were evaluated using three different classification thresholds.

| Threshold | Precision | Recall | F1-Score | Accuracy |
|---|---:|---:|---:|---:|
| 0.50 | 1.000 | 0.762 | 0.865 | 0.975 |
| 0.40 | 1.000 | 0.857 | 0.923 | 0.985 |
| 0.30 | 1.000 | 0.857 | 0.923 | 0.985 |

### Selected Threshold: 0.40

A threshold of 0.40 was selected because:

- Recall improved from 0.762 to 0.857.
- Precision remained at 1.000.
- F1-Score improved from 0.865 to 0.923.
- Accuracy improved from 0.975 to 0.985.
- Reducing the threshold further to 0.30 did not provide an additional improvement in these results.

Therefore, 0.40 was selected as the preferred threshold for this experiment.

The threshold is not universally optimal; it should be selected according to the real-world cost of false positives and false negatives.

## 7. Cost of False Positives and False Negatives

### Example: Student Academic Support System

Consider a system that predicts whether a student may be at academic risk.

**False Positive (FP):**
- A student is predicted to be at risk but is actually not at risk.
- This may result in unnecessary counselling or additional staff effort.

**False Negative (FN):**
- A student is actually at risk but the model predicts that they are not.
- The student may miss timely academic support.

In this use case, false negatives may be more costly because students who need support could be overlooked. Therefore, recall is an important metric.

However, precision must also be considered because too many false alarms can increase the workload of support staff.

## 8. Final Conclusion

In this practical, Logistic Regression was evaluated using different classification metrics on an imbalanced dataset.

The experiment demonstrated that accuracy alone is not sufficient for evaluating an imbalanced classification problem. Precision, recall, F1-score, ROC-AUC, and PR-AUC provide additional insights into model performance.

Class weighting improved minority-class recall, while threshold selection provided another way to balance precision and recall.

A threshold of 0.40 gave the best F1-score and accuracy among the tested thresholds in this experiment.

Thus, the choice of evaluation metric, class weighting, and classification threshold should depend on the problem and the cost of incorrect predictions.

## 9. Tools and Libraries Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 10. Output Files

- `Day_08_Metrics_Imbalanced_Data.ipynb` – Practical notebook
- `metrics_comparison.csv` – Comparison of baseline and class-weighted models
- `threshold_comparison.csv` – Comparison of different classification thresholds
- `README.md` – Practical report and documentation


