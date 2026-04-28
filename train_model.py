import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

def train_and_save_model(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found. Please ensure the path is correct.")

    df1 = pd.read_csv(dataset_path)

    # Figure out target column
    if 'default' in df1.columns:
        y = 1 - df1['default']  # 1 means Paid Back (repay) and 0 means Default
        X = df1.drop('default', axis=1)
    elif 'loan_paid_back' in df1.columns:
        y = df1['loan_paid_back']
        X = df1.drop('loan_paid_back', axis=1)
    else:
        raise ValueError("No recognized target column ('default' or 'loan_paid_back') found.")

    # Drop useless columns
    drop_cols = ['customer_id', 'person_id', 'person_name']
    X = X.drop(columns=[col for col in drop_cols if col in X.columns])

    # Handle categorical features dynamically
    cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade', 'employment_type']
    cat_cols = [c for c in cat_cols if c in X.columns]

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    
    # Drop remaining object/string columns that we couldn't encode
    obj_cols = X_encoded.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        X_encoded = X_encoded.drop(columns=obj_cols)

    # Impute NaNs to avoid scaling errors
    X_encoded = X_encoded.fillna(0)

    final_columns = list(X_encoded.columns)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Model training
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)

    # Save pipeline assets
    with open("scaler_rf.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("model_rf.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("columns_rf.pkl", "wb") as f:
        pickle.dump(final_columns, f)

    return True

if __name__ == '__main__':
    train_and_save_model('uploads/LoanSightAI_loan_dataset.csv')
    print("--- PREPROCESSING PIPELINE SUCCESSFUL ---")
