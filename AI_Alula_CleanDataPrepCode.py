import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset with anomalies
df = pd.read_excel("AI_Alula_Dataset_With_Anomalies.xlsx")

# -------------------------
# STEP 1: Fix Format Issues
# -------------------------
# Convert to numeric, coercing errors (invalid formats like "forty" → NaN)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Blood_Marker_2'] = pd.to_numeric(df['Blood_Marker_2'], errors='coerce')

# --------------------------
# STEP 2: Remove Duplicates
# --------------------------
df.drop_duplicates(inplace=True)

# -----------------------------------
# STEP 3: Handle Missing Values (Central Tendency)
# -----------------------------------
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Blood_Marker_1'].fillna(df['Blood_Marker_1'].median(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# -----------------------------------
# STEP 4: Outlier Removal Using IQR
# -----------------------------------
def remove_iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_iqr_outliers(df, 'Age')
df = remove_iqr_outliers(df, 'Blood_Marker_1')

# ------------------------------------
# STEP 5: Outlier Removal Using Z-score
# ------------------------------------
z_scores = np.abs(stats.zscore(df['Blood_Marker_2'].dropna()))
threshold = 3
valid_indices = df['Blood_Marker_2'].dropna().index[z_scores < threshold]
df = df.loc[valid_indices]

# ------------------------------------
# STEP 6: Finalize
# ------------------------------------
df.reset_index(drop=True, inplace=True)

# Save the cleaned dataset
df.to_excel("AI_Alula_CleanedDataset.xlsx", index=False)

print("✅ Cleaned dataset saved as 'AI_Alula_CleanedDataset.xlsx'")
