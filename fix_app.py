import re

with open('app.py', 'r') as f:
    code = f.read()

# Remove tensorflow import
code = re.sub(r'import tensorflow as tf\n', '', code)

# Update load_ml_assets
new_load_ml = """def load_ml_assets():
    global model, scaler, features_list, is_model_ready
    try:
        with open('model_rf.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler_rf.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('columns_rf.pkl', 'rb') as f:
            features_list = pickle.load(f)
        is_model_ready = True
        return True
    except Exception as e:
        print(f"Error loading assets: {e}")
        return False"""
code = re.sub(r'def load_ml_assets\(\):.*?(?=@app\.route)', new_load_ml + '\n\n', code, flags=re.DOTALL)

# Update predict method inside POST
predict_post = """        full_data_str = request.form.get('full_data', '{}')
        try:
            full_data = json.loads(full_data_str)
        except:
            full_data = {}
            
        # Hard validation on explicit vars
        income = float(request.form.get('income', 0))
        loan_amount = float(request.form.get('loan_amount', 0))
        credit_score = float(request.form.get('credit_score', 0))
        employment_years = float(request.form.get('employment_years', 2))
        
        if not (300 <= credit_score <= 900):
            return "Error: Credit score must be between 300 and 900. Go back and alter values.", 400
            
        # Compute derived features
        loan_income_ratio = loan_amount / income if income > 0 else 0
        
        form_data = {
            'income': income,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'loan_income_ratio': loan_income_ratio,
            'employment_years': employment_years
        }
        
        full_data.update(form_data)
        
        # update underlying annual_income if it exists
        if 'annual_income' in full_data:
            full_data['annual_income'] = income
            
        person_name = request.form.get('person_name', 'Unknown')
        person_id = request.form.get('person_id', 'N/A')
        
        form_data['person_name'] = person_name
        form_data['person_id'] = person_id
        
        full_data['person_name'] = person_name
        full_data['person_id'] = person_id

        # Print Debug Log
        print(f"\\n--- LOANSIGHT AI INFERENCE ---")
        
        # Preprocess full_data for Random Forest
        import pandas as pd
        df_infer = pd.DataFrame([full_data])
        
        # Handle derived column just in case model expects it
        if 'loan_income_ratio' in features_list and 'loan_income_ratio' not in df_infer.columns:
            df_infer['loan_income_ratio'] = loan_income_ratio
            
        drop_cols = ['person_id', 'person_name', 'loan_paid_back']
        df_infer = df_infer.drop(columns=[col for col in drop_cols if col in df_infer.columns])
        
        cat_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
        cat_cols = [c for c in cat_cols if c in df_infer.columns]
        df_encoded = pd.get_dummies(df_infer, columns=cat_cols)
        
        for col in features_list:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[features_list]
        
        scaled_data = scaler.transform(df_encoded)
        
        repayment_probability = float(model.predict_proba(scaled_data)[0][1])
        default_probability = 1.0 - repayment_probability"""
code = re.sub(r'        full_data_str = request\.form\.get.*?default_probability = 1\.0 - raw_prediction', predict_post, code, flags=re.DOTALL)

# Update display_features in predict GET
display_features = "    display_features = ['income', 'loan_amount', 'credit_score', 'employment_years']"
code = re.sub(r'    display_features = \[f for f in features_list if f != \'loan_income_ratio\'\]', display_features, code)

with open('app.py', 'w') as f:
    f.write(code)
print("app.py updated successfully")
