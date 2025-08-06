import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of synthetic patients
num_records = 1000

# Generate Patient IDs
patient_ids = [f"PID{1000+i}" for i in range(num_records)]

# Generate Ages (normal distribution, clipped)
ages = np.clip(np.random.normal(loc=50, scale=15, size=num_records), 18, 90).astype(int)

# Generate Gender (M/F)
genders = np.random.choice(['Male', 'Female'], size=num_records)

# Generate Smoking Status (Yes/No)
smoking_status = np.random.choice(['Yes', 'No'], size=num_records, p=[0.3, 0.7])

# Generate Alcohol Use (Yes/No)
alcohol_use = np.random.choice(['Yes', 'No'], size=num_records, p=[0.4, 0.6])

# Family History of Cancer (Yes/No)
family_history = np.random.choice(['Yes', 'No'], size=num_records, p=[0.2, 0.8])

# Blood Marker 1 (biomarker levels: normal range 0-10, cancerous >7)
marker1 = np.round(np.random.normal(loc=5, scale=2, size=num_records), 2)

# Blood Marker 2 (normal 50-150, cancerous >130)
marker2 = np.round(np.random.normal(loc=100, scale=20, size=num_records), 1)

# Symptom Score (0-10)
symptom_score = np.random.randint(0, 11, size=num_records)

# Target: Cancer Diagnosis (0 = No, 1 = Yes)
# We base it on a simple rule: marker1 > 7 and marker2 > 130 → more likely to have cancer
diagnosis = []
for i in range(num_records):
    if marker1[i] > 7 and marker2[i] > 130:
        diagnosis.append(1 if random.random() > 0.2 else 0)  # 80% chance of having cancer
    else:
        diagnosis.append(0 if random.random() > 0.2 else 1)  # 20% false positive rate

# Create DataFrame
df = pd.DataFrame({
    'Patient_ID': patient_ids,
    'Age': ages,
    'Gender': genders,
    'Smoking_Status': smoking_status,
    'Alcohol_Use': alcohol_use,
    'Family_History': family_history,
    'Blood_Marker_1': marker1,
    'Blood_Marker_2': marker2,
    'Symptom_Score': symptom_score,
    'Cancer_Diagnosis': diagnosis
})

# Preview the data
print(df.head())

# Save to Excel
df.to_excel("AI_Alula_CleanedDataset.xlsx", index=False)
print("Synthetic dataset saved to AI_Alula_CleanedDataset.xlsx")