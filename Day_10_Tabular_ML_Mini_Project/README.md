\# Day 10: Tabular ML Mini-Project



\## 1. Project Overview



This project packages a complete tabular machine learning workflow using a student performance dataset.



It demonstrates:

\- Loading and preparing tabular data

\- Separating features and target variables

\- Splitting data into training and testing sets

\- Handling missing numerical and categorical values

\- Scaling numerical features and encoding categorical features

\- Training a Logistic Regression model

\- Evaluating model performance

\- Saving the trained model and evaluation reports

\- Running automated tests



\## 2. Project Structure



```text

Day\_10\_Tabular\_ML\_Mini\_Project/

│

├── config/

│   └── config.json

│

├── data/

│   └── student\_performance.csv

│

├── models/

│   └── student\_performance\_model.joblib

│

├── reports/

│   ├── metrics.csv

│   └── classification\_report.txt

│

├── src/

│   └── train.py

│

├── tests/

│   └── test\_model.py

│

├── requirements.txt

├── run\_project.py

└── README.md

```



\## 3. Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- Pytest



\## 4. How to Run the Project



\### Step 1: Install dependencies



```bash

python -m pip install -r requirements.txt

```



\### Step 2: Run the complete workflow



```bash

python run\_project.py

```



The command automatically:

1\. Trains the machine learning model.

2\. Evaluates its performance.

3\. Saves the trained model and reports.

4\. Runs automated tests.



\## 5. Model Evaluation



The model was evaluated using a test set containing 5 records.



| Metric | Result |

|---|---:|

| Accuracy | 1.0000 |

| Precision | 1.0000 |

| Recall | 1.0000 |

| F1-score | 1.0000 |



\## 6. Automated Testing



Four automated tests were executed successfully:



\- Verify that the saved model exists.

\- Verify that the saved model loads correctly.

\- Verify that the model generates predictions.

\- Verify that predictions contain valid class labels.



\*\*Test result: 4 passed.\*\*



\## 7. Model Persistence



The trained machine learning pipeline is saved using Joblib.



File: `models/student\_performance\_model.joblib`



This allows the trained pipeline to be loaded and reused without training it again.



Only load model files from trusted sources.



\## 8. Limitations



\- The dataset contains only 24 records.

\- The test set contains only 5 records.

\- The evaluation results may not represent performance on larger, unseen datasets.

\- The model requires validation using more diverse and independent data before real-world use.



\## 9. Future Improvements



\- Use a larger dataset.

\- Perform cross-validation.

\- Compare additional machine learning algorithms.

\- Evaluate the model on independent data.

\- Improve configuration and input validation.



\## 10. Conclusion



This project demonstrates a reproducible tabular machine learning workflow, from data preparation and model training to evaluation, model persistence, and automated testing.

