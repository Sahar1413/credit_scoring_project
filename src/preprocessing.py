"""
📌 Preprocessing script for German Credit Data
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Load dataset
def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "german.data")
    column_names = [
        "Status_Checking", "Duration", "Credit_History", "Purpose", "Credit_Amount",
        "Savings", "Employment", "Installment_Rate", "Personal_Status", "Other_Debtors",
        "Residence_Since", "Property", "Age", "Other_Installment_Plans", "Housing",
        "Existing_Credits", "Job", "Liable_People", "Telephone", "Foreign_Worker", "Target"
    ]
    data = pd.read_csv(path, sep=" ", header=None, names=column_names)
    return data

# Split features & target
def split_features_target(data):
    X = data.drop("Target", axis=1)
    y = data["Target"].apply(lambda x: 0 if x == 1 else 1)  # 0 = Good, 1 = Bad
    return X, y

# Build preprocessing pipeline
def build_preprocessing_pipeline(X):
    categorical_features = X.select_dtypes(include=["object"]).columns
    numeric_features = X.select_dtypes(exclude=["object"]).columns

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features)
        ]
    )
    return preprocessor

# Save train/test split
def prepare_data(test_size=0.2, random_state=42):
    data = load_data()
    X, y = split_features_target(data)
    preprocessor = build_preprocessing_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print("✅ Data preprocessing complete")
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
