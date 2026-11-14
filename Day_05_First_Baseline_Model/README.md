
# Day 5 – First Baseline Model

## Objective
Train a classification model and establish a reproducible baseline using Scikit-learn.

## Dataset
**Handwritten Digits Dataset**

- Source: Scikit-learn built-in dataset
- Total samples: 1,797
- Features: 64 pixel values
- Target: Digit labels (0–9)
- Number of classes: 10

## Features and Target
- X (Features): 64 pixel intensity values.
- y (Target): Handwritten digit labels (0–9).

## Data Preparation
- Training data: 80% (1,437 samples)
- Testing data: 20% (360 samples)
- Random state: 42
- Stratified train-test split was used.

## Models Implemented

### 1. DummyClassifier
- Strategy: Most Frequent
- Used as the baseline model.

### 2. Logistic Regression
- Used StandardScaler for preprocessing.
- Implemented using Scikit-learn Pipeline.
- Maximum iterations: 2000
- Random state: 42

## Evaluation Results

| Model | Accuracy |
|---|---:|
| DummyClassifier | 10.00% |
| Logistic Regression | 97.22% |

Additional metrics (Precision, Recall and F1-score) are saved in `baseline_metrics.csv`.

## Conclusion
Logistic Regression achieved 97.22% accuracy, significantly outperforming the DummyClassifier baseline of 10%.

The experiment demonstrates how a classification model can learn meaningful patterns from handwritten digit images.

No hyperparameter tuning was performed. The results represent the initial baseline experiment.

## Files
- baseline_model.ipynb – Complete implementation.
- baseline_metrics.csv – Saved evaluation metrics.
- README.md – Project documentation.

