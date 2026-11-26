
import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ----------------------------------
# PROJECT PATHS
# ----------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_DIR / "config" / "config.json"

# Load configuration
with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

DATA_PATH = PROJECT_DIR / config["dataset_path"]

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# ----------------------------------
# LOAD DATA
# ----------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ----------------------------------
# SEPARATE FEATURES AND TARGET
# ----------------------------------

target_column = config["target_column"]
id_column = config["id_column"]

X = df.drop(columns=[target_column, id_column])
y = df[target_column]

numerical_features = config["numerical_features"]
categorical_features = config["categorical_features"]


# ----------------------------------
# TRAIN-TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["test_size"],
    random_state=config["random_state"],
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ----------------------------------
# PREPROCESSING
# ----------------------------------

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# ----------------------------------
# BUILD MODEL PIPELINE
# ----------------------------------

model = LogisticRegression(
    max_iter=config["model"]["max_iter"]
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])


# ----------------------------------
# TRAIN MODEL
# ----------------------------------

print("\nTraining model...")

model_pipeline.fit(X_train, y_train)

print("Model training completed!")


# ----------------------------------
# MODEL EVALUATION
# ----------------------------------

y_pred = model_pipeline.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(
        y_test, y_pred, pos_label="Pass", zero_division=0
    ),
    "recall": recall_score(
        y_test, y_pred, pos_label="Pass", zero_division=0
    ),
    "f1_score": f1_score(
        y_test, y_pred, pos_label="Pass", zero_division=0
    )
}

print("\nMODEL EVALUATION")
print("-------------------------")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


# ----------------------------------
# SAVE EVALUATION REPORT
# ----------------------------------

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(
    REPORT_DIR / "metrics.csv",
    index=False
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

with open(REPORT_DIR / "classification_report.txt", "w") as file:
    file.write(report)


# ----------------------------------
# SAVE TRAINED MODEL
# ----------------------------------

joblib.dump(
    model_pipeline,
    MODEL_DIR / "student_performance_model.joblib"
)

print("\nModel saved successfully!")
print("Metrics saved successfully!")

print("\nDay 10 training workflow completed!")

