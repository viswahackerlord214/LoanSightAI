import pandas as pd
import numpy as np

def regenerate_realistic_data(file_path):
    df = pd.read_csv(file_path)

    np.random.seed(42)

    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
    df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
    
    risk_score = np.zeros(len(df))
    
    risk_score += np.where(df['debt_to_income_ratio'] > 0.40, 8, 
                  np.where(df['debt_to_income_ratio'] > 0.30, 4, 
                  np.where(df['debt_to_income_ratio'] < 0.20, -3, 0)))
                  
    risk_score += np.where(df['credit_score'] < 620, 8, 
                  np.where(df['credit_score'] < 680, 4, 
                  np.where(df['credit_score'] > 720, -4, 0)))
                  
    risk_score += df['late_payments_last_2yrs'] * 4
    
    risk_score -= np.where(df['annual_income'] > df['loan_amount'], 4, 0)
    risk_score += np.where(df['loan_amount'] > df['annual_income'] * 2.5, 6, 0)
    
    probabilities = 1 / (1 + np.exp(-(risk_score - 4) * 2.0))
    probabilities = np.clip(probabilities, 0.005, 0.995)
    
    new_defaults = (np.random.random(len(df)) < probabilities).astype(int)
    
    df['default'] = new_defaults
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    regenerate_realistic_data('uploads/LoanSightAI_loan_dataset.csv')
