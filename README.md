# 🏦 LoanSight AI

An intelligent loan risk assessment platform that leverages Machine Learning to evaluate loan applicants, predict repayment probability, and generate professional risk assessment reports.

---

## 📌 Overview

LoanSight AI is a web-based loan analytics platform designed to assist financial institutions, lenders, and analysts in making data-driven lending decisions.

The system allows users to upload loan datasets, search for customers, modify financial parameters in real time, and instantly obtain repayment risk predictions using a dynamic 3-tier risk logic system, along with downloadable PDF reports.

---

## 🚀 Features

### 📂 Dataset Ingestion & Dynamic Training

* Upload custom loan datasets in CSV format.
* Automatic data cleaning and preprocessing on upload.
* Dynamic handling of categorical and numerical features.
* The Random Forest model is completely retrained dynamically every time a new dataset is uploaded.
* Automatically aligns web form inputs with variable columns dynamically.

---

### 🔍 Customer Discovery

* Search applicants using:
  * Customer Name (`person_name`, `name`)
  * Customer ID (`person_id`, `customer_id`)
* Retrieves complete customer financial profiles instantly from the uploaded dataset.
* Seamlessly auto-fills the prediction form with a single click.

---

### 🤖 AI-Powered 3-Tier Risk Prediction

Users can input or prefill key financial metrics such as:
* Annual Income
* Loan Amount
* Credit Score Proxy
* Employment Experience

The system automatically:
* Performs intelligent feature engineering and mapping to align standard form inputs with dynamic dummy-encoded dataset variables.
* Processes data through the trained Random Forest pipeline.
* Generates real-time probabilistic predictions using `predict_proba()`.

#### 3-Tier Prediction Output

* 🟢 **Likely to Repay (Low Risk):** Default probability < 30%
* 🟡 **Needs Review (Moderate Risk):** Default probability 30% - 60%
* 🔴 **High Default Risk (High Risk):** Default probability > 60%

---

### 📄 Professional PDF Reporting

Generate downloadable reports containing:
* Customer Details
* Financial Information
* Prediction Results
* AI Summary
* Risk Indicators

Reports are generated using ReportLab for professional formatting.

---

## 🏗 System Architecture

```text
CSV Dataset
      │
      ▼
Data Cleaning & Preprocessing
      │
      ▼
Feature Engineering (Dynamic Alignment)
      │
      ▼
One-Hot Encoding (Dummy Variables)
      │
      ▼
Feature Scaling
      │
      ▼
Random Forest Model (Dynamic Retraining)
      │
      ▼
Risk Prediction (3-Tier Probability)
      │
      ▼
PDF Report Generation
```

---

## 🛠 Technology Stack

### Backend
* Flask
* Flask Sessions
* Python

### Machine Learning
* Scikit-Learn
* Random Forest Classifier
* StandardScaler
* Pandas
* NumPy
* Pickle

### Frontend
* HTML5
* Jinja2
* Vanilla CSS (Rich Custom Glassmorphism UI)
* JavaScript
* Chart.js (Real-time Probability Charts)

### Reporting
* ReportLab

---

## 🧠 Machine Learning Pipeline

### 1. Data Ingestion
The system loads the user-uploaded dataset:
```text
uploads/filename.csv
```

### 2. Feature Encoding
Categorical variables are transformed using Pandas One-Hot Encoding (`pd.get_dummies`):
* employment_type
* loan_purpose
* person_home_ownership
* loan_intent
* loan_grade
* cb_person_default_on_file

### 3. Feature Alignment
The system automatically bridges standard web-form inputs to match the precise engineered columns expected by the model.

### 4. Feature Scaling
Numerical variables are standardized using `StandardScaler` to ensure consistent feature distributions during training and inference.

### 5. Model Training
Algorithm:
```python
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
```
Benefits:
* Handles tabular data effectively
* Robust against overfitting
* Strong performance on mixed feature types
* Provides reliable probability estimates using `predict_proba()`

### 6. Model Serialization
Trained artifacts are stored using Pickle:
```text
model_rf.pkl
scaler_rf.pkl
columns_rf.pkl
```

---

## 📁 Project Structure

```text
LoanSight-AI/
│
├── app.py
├── train_model.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── input.html
│   ├── result.html
│   ├── search.html
│   └── upload.html
│
├── uploads/
│
├── model_rf.pkl
├── scaler_rf.pkl
├── columns_rf.pkl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/viswahackerlord214/LoanSightAI.git
cd LoanSightAI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows
```bash
venv\Scripts\activate
```

#### macOS/Linux
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:
```text
http://127.0.0.1:5000
```

---

## 📊 Sample Workflow

1. Upload a loan dataset via the web UI.
2. The Random Forest model automatically trains and saves itself.
3. Search a customer by ID (e.g. CUST00923) or Name.
4. Auto-fill the customer profile into the prediction form.
5. Run AI prediction.
6. Analyze repayment probability in the 3-Tier risk chart.
7. Download PDF assessment report.

---

## 👨‍💻 Author

**Viswa Ravindren**

B.Tech Computer Science Engineering  
Motilal Nehru National Institute of Technology (MNNIT)

---

## 📜 License

This project is developed for educational, research, and portfolio purposes.
