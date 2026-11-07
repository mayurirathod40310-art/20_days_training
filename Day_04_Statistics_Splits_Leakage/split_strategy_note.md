# Day 4: Train-Test Split Strategy Note

## 1. Objective
To understand statistical measures and design a valid train-test splitting strategy.

## 2. Dataset Information
- Total records: 20
- Features: Study_Hours, Attendance
- Target column: Result
- Classes: Pass and Fail

## 3. Statistical Analysis

| Measure | Study Hours | Attendance |
|---|---:|---:|
| Mean | 4.0 | 70.4 |
| Median | 4.0 | 70.0 |
| Variance | 2.63 | 126.88 |

## 4. Class Distribution
- Pass: 12 records (60%)
- Fail: 8 records (40%)

## 5. Splitting Strategy
Used an 80:20 train-test split with stratified sampling.

- Training records: 16
- Testing records: 4
- random_state = 42
- stratify = y

Stratified sampling helps maintain class proportions across training and testing datasets.

## 6. Reproducibility
The random_state parameter was set to 42 to ensure consistent splitting results across repeated executions.

## 7. Conclusion
The dataset was successfully analyzed and divided into training and testing sets using stratified sampling. This provides a consistent basis for evaluating a machine learning model.