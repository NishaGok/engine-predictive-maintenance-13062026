
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# --------------------------------------------------
# Hugging Face Setup
# --------------------------------------------------

api = HfApi(
    token=os.getenv("HF_TOKEN1")
)

DATASET_PATH = (
    "hf://datasets/"
    "NishaGok/engine-predictive-maintenance-13062026/"
    "engine_data.csv"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully")
print("Dataset Shape:", df.shape)

# --------------------------------------------------
# Standardize Column Names
# Same as notebook
# --------------------------------------------------

df.columns = (
    df.columns
      .str.strip()
      .str.replace(" ", "_")
      .str.lower()
)

print("Columns standardized")

# --------------------------------------------------
# Data Quality Checks
# --------------------------------------------------

print("Missing Values:")
print(df.isnull().sum())

print("Duplicate Rows:")
print(df.duplicated().sum())

# --------------------------------------------------
# Target Variable
# --------------------------------------------------

TARGET_COL = "engine_condition"

X = df.drop(columns=[TARGET_COL])

y = df[TARGET_COL]

# --------------------------------------------------
# Train Test Split
# Same as notebook
# --------------------------------------------------

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train Shape:", Xtrain.shape)
print("Test Shape:", Xtest.shape)

# --------------------------------------------------
# Save Files
# --------------------------------------------------

Xtrain.to_csv(
    "Xtrain.csv",
    index=False
)

Xtest.to_csv(
    "Xtest.csv",
    index=False
)

ytrain.to_csv(
    "ytrain.csv",
    index=False
)

ytest.to_csv(
    "ytest.csv",
    index=False
)

print("Train/Test files saved")

# --------------------------------------------------
# Upload Files to HF Dataset Repo
# --------------------------------------------------

files = [
    "Xtrain.csv",
    "Xtest.csv",
    "ytrain.csv",
    "ytest.csv"
]

for file_path in files:

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="NishaGok/engine-predictive-maintenance-13062026",
        repo_type="dataset"
    )

    print(f"Uploaded {file_path}")

print("All train/test files uploaded successfully")
