import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Create features.json including derived feature
features = ["income", "loan_amount", "credit_score", "loan_income_ratio", "employment_years"]
with open("features.json", "w") as f:
    json.dump(features, f)

# 2. Generate Realistic Dummy Data
# Class 1: Repaint (High Income, Low Loan, High Credit)
# Class 0: Default (Low Income, High Loan, Low Credit)
np.random.seed(42)

# Generate 500 records of people who WILL repay
repays = np.random.normal(loc=[100000, 50000, 750, 0.5, 6], scale=[20000, 10000, 50, 0.2, 3], size=(500, 5))
y_repays = np.ones(500)

# Generate 500 records of people who will NOT repay (High Risk)
defaults = np.random.normal(loc=[30000, 200000, 500, 6.6, 2], scale=[10000, 50000, 80, 2, 1], size=(500, 5))
y_defaults = np.zeros(500)

X_data = np.vstack((repays, defaults))
y_data = np.concatenate((y_repays, y_defaults))

# Ensure no negative scaling artifacts
X_data[X_data < 0] = 1

# 3. Fit scaler.pkl
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 4. Train RandomForest w/ Balanced Classes
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_scaled, y_data)

# Save the sklearn model directly
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Realistic Scikit-Learn Random Forest Pipeline + Scaler generated successfully!")
