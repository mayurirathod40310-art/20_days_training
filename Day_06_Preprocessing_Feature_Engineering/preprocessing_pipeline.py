import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("student_performance.csv")

# Separate features and target
X = df.drop(columns=["Student_ID", "Result"])
y = df["Result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numerical_features = ["Age", "Study_Hours", "Attendance"]
categorical_features = ["Department"]


# --------------------------------
# MODEL 1: MANUAL BASELINE
# --------------------------------

# Copy datasets
X_train_base = X_train.copy()
X_test_base = X_test.copy()

# Fill numerical missing values using training medians
for col in numerical_features:
    median_value = X_train_base[col].median()
    X_train_base[col] = X_train_base[col].fillna(median_value)
    X_test_base[col] = X_test_base[col].fillna(median_value)

# Fill categorical missing values using training mode
for col in categorical_features:
    mode_value = X_train_base[col].mode()[0]
    X_train_base[col] = X_train_base[col].fillna(mode_value)
    X_test_base[col] = X_test_base[col].fillna(mode_value)

# One-hot encoding
X_train_base = pd.get_dummies(X_train_base, dtype=int)
X_test_base = pd.get_dummies(X_test_base, dtype=int)

# Align testing columns with training columns
X_test_base = X_test_base.reindex(
    columns=X_train_base.columns,
    fill_value=0
)

# Train baseline model
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train_base, y_train)

# Predictions
baseline_pred = baseline_model.predict(X_test_base)

baseline_accuracy = accuracy_score(y_test, baseline_pred)


# --------------------------------
# MODEL 2: PIPELINE-BASED MODEL
# --------------------------------

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Complete pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train pipeline model
model_pipeline.fit(X_train, y_train)

# Save the complete trained pipeline
joblib.dump(model_pipeline, "student_result_pipeline.joblib")

print("\nReusable pipeline saved successfully!")

# Predictions
pipeline_pred = model_pipeline.predict(X_test)

pipeline_accuracy = accuracy_score(y_test, pipeline_pred)


# --------------------------------
# COMPARE BOTH MODELS
# --------------------------------

print("\nMODEL COMPARISON")
print("---------------------------")

print("Baseline Accuracy:", baseline_accuracy)
print("Pipeline Accuracy:", pipeline_accuracy)

print("\nComparison Table:")

comparison = pd.DataFrame({
    "Model": ["Manual Baseline", "Pipeline Model"],
    "Accuracy": [baseline_accuracy, pipeline_accuracy]
})

print(comparison)

print("\nDay 6 Model Comparison Completed!")


