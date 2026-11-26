
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

print("\n===================================")
print(" DAY 10 - TABULAR ML MINI PROJECT")
print("===================================")

# Step 1: Train and evaluate the model
print("\nSTEP 1: MODEL TRAINING")

subprocess.run(
    [sys.executable, "src/train.py"],
    cwd=PROJECT_DIR,
    check=True
)

# Step 2: Run automated tests
print("\nSTEP 2: AUTOMATED TESTING")

subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "-v"],
    cwd=PROJECT_DIR,
    check=True
)

print("\n===================================")
print(" PROJECT EXECUTION COMPLETED!")
print("===================================")

