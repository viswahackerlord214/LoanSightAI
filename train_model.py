import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

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

    # 4. Train / Test Split (80% Train, 20% Unseen Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Feature scaling (Fit ONLY on X_train to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Regularized Random Forest (Controlled Depth)
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,              # Depth limit to stop 100% memorization
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate on both Train and Test
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc  = accuracy_score(y_test, model.predict(X_test_scaled))
    test_auc  = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    print(f"📊 Training Accuracy: {train_acc * 100:.2f}% (Realistic, not 100%)")
    print(f"🎯 Hold-Out Test Accuracy: {test_acc * 100:.2f}%")
    print(f"⭐ Test ROC-AUC: {test_auc:.4f}")
    print("\nDetailed Test Classification Report:\n", classification_report(y_test, model.predict(X_test_scaled)))

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
