
import joblib
import pandas as pd

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "student_performance_model.joblib"
DATA_PATH = PROJECT_DIR / "data" / "student_performance.csv"


def test_saved_model_exists():
    assert MODEL_PATH.exists()


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_predictions():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Student_ID", "Result"])

    predictions = model.predict(X)

    assert len(predictions) == len(df)


def test_prediction_labels():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Student_ID", "Result"])

    predictions = model.predict(X)

    assert set(predictions).issubset({"Pass", "Fail"})

