import pandas as pd
import joblib

# Load the saved pipeline
model = joblib.load("student_result_pipeline.joblib")

# Create new student records
new_students = pd.DataFrame({
    "Age": [21, None],
    "Study_Hours": [5, 2],
    "Attendance": [85, 55],
    "Department": ["CSE", None]
})

# Make predictions
predictions = model.predict(new_students)

# Display predictions
print("New Student Records:")
print(new_students)

print("\nPredicted Results:")
for i, prediction in enumerate(predictions, start=1):
    print(f"Student {i}: {prediction}")

print("\nSaved pipeline tested successfully!")
