

\# Day 6: Preprocessing and Feature Engineering



\## Objective

Build a reusable preprocessing pipeline using Scikit-learn and compare a manually preprocessed baseline model with a pipeline-based model.



\## Dataset

The project uses a Student Performance dataset containing 24 records.



Features:

\- Age

\- Study\_Hours

\- Attendance

\- Department



Target:

\- Result (Pass/Fail)



\## Tasks Performed



1\. Loaded and explored the dataset using Pandas.

2\. Identified and handled missing values.

3\. Separated input features and target variable.

4\. Split data into training and testing sets (80:20).

5\. Applied median imputation to numerical features.

6\. Applied most-frequent imputation to categorical features.

7\. Used StandardScaler for numerical scaling.

8\. Applied OneHotEncoder for categorical encoding.

9\. Combined preprocessing steps using ColumnTransformer.

10\. Built a complete machine learning Pipeline using Logistic Regression.

11\. Compared manual baseline and pipeline-based model accuracy.

12\. Saved and tested the reusable pipeline using Joblib.



\## Model Comparison



| Model | Accuracy |

|---|---:|

| Manual Baseline | 100% |

| Pipeline Model | 100% |



Note: Accuracy is based on only 5 test records and should not be interpreted as evidence of real-world performance.



\## Project Files



\- `student\_performance.csv` – Dataset

\- `preprocessing\_pipeline.py` – Main implementation

\- `test\_saved\_pipeline.py` – Tests the saved pipeline

\- `student\_result\_pipeline.joblib` – Saved trained pipeline



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib



\## Learning Outcomes



\- Understanding data preprocessing and feature engineering.

\- Handling missing values through imputation.

\- Applying numerical scaling and categorical encoding.

\- Preventing data leakage using train-test splitting.

\- Building reusable preprocessing pipelines.

\- Saving and loading trained machine learning pipelines.



\## Conclusion



Successfully developed and tested a reusable Scikit-learn preprocessing pipeline that combines data imputation, scaling, categorical encoding, and Logistic Regression classification.





