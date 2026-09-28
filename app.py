"""
LoanSight AI — Flask Backend
IMPORTANT: ML model logic is NOT modified.
Only UI/UX routes, state management, search, and PDF generation are added/updated.
"""

import os
import io
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, session, send_file)

from train_model import train_and_save_model

# ── PDF ──────────────────────────────────────────────────────────────────────
# Use reportlab if available; fallback to fpdf2
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    PDF_ENGINE = "reportlab"
except ImportError:
    try:
        from fpdf import FPDF
        PDF_ENGINE = "fpdf"
    except ImportError:
        PDF_ENGINE = None

# ── TensorFlow / Keras ───────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    # from tensorflow.keras.models import load_model
    MODEL_FRAMEWORK = None
except ImportError:
    MODEL_FRAMEWORK = None

# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "loansight-secret-2026"   # Change in production

# ── Global state (simple in-memory; use a DB for production) ─────────────────
is_model_trained = False
uploaded_df = None
_features_list = None

MODEL_PATH  = "model_rf.pkl"
SCALER_PATH = "scaler_rf.pkl"
COLUMNS_PATH = "columns_rf.pkl"

# Load pre-existing model/scaler at startup (if files already exist)
try:
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        _features_list = pickle.load(f)
    is_model_trained = True
    print("[LoanSight] Pre-trained Random Forest model and scaler loaded from disk.")
except Exception as e:
    _model  = None
    _scaler = None
    _features_list = None
    print(f"[LoanSight] No pre-trained model found ({e}). Upload a dataset first.")

# ── Feature columns expected by the model (edit to match your actual features) ─
FEATURE_COLS = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file", "loan_status"
]

# ── Categorical mappings (adjust to match your training encoding) ─────────────
HOME_MAP    = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
INTENT_MAP  = {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2,
               "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5}
GRADE_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
DEFAULT_MAP = {"N": 0, "Y": 1}


# ── Helper: inject is_model_trained into every template ──────────────────────
@app.context_processor
def inject_globals():
    return dict(is_model_trained=is_model_trained, now=datetime.now().strftime("%d %b %Y, %H:%M"))


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── UPLOAD & BATCH PREDICT ───────────────────────────────────────────────────
@app.route("/upload", methods=["GET", "POST"])
def upload():
    global is_model_trained, uploaded_df, _model, _scaler, _features_list

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.", "error")
            return redirect(url_for("upload"))

        try:
            # Save file
            os.makedirs("uploads", exist_ok=True)
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)
            
            df = pd.read_csv(filepath)
            
            # --- Batch Inference ---
            if _model is not None and _features_list is not None:
                print(f"[LoanSight] Running batch inference on {len(df)} applicants...")
                df_infer = df.copy()
                
                # Column normalization (e.g. from new_loan_dataset vs old)
                if 'loan_term' in df_infer.columns and 'loan_term_months' not in df_infer.columns:
                    df_infer['loan_term_months'] = df_infer['loan_term']
                
                # Encode categorical features
                cat_cols = ['employment_type', 'loan_purpose']
                cat_cols = [c for c in cat_cols if c in df_infer.columns]
                if cat_cols:
                    df_encoded = pd.get_dummies(df_infer, columns=cat_cols)
                else:
                    df_encoded = df_infer
                
                # Align with training features
                for col in _features_list:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                
                # Ensure correct order and fill NaNs
                X_df = df_encoded[_features_list].fillna(0)
                
                # Scale and predict
                X_scaled = _scaler.transform(X_df)
                probs = _model.predict_proba(X_scaled)[:, 1] # Probability of repayment
                
                # Compute risk
                df['repayment_probability'] = np.round(probs * 100, 2)
                df['default_probability'] = 1.0 - probs
                
                def get_tier(prob_def):
                    if prob_def < 0.30: return "Likely to Repay"
                    if prob_def <= 0.60: return "Needs Review"
                    return "High Default Risk"
                
                df['predicted_risk_tier'] = df['default_probability'].apply(get_tier)
                
                flash(f"✅ Successfully scored {len(df)} applicants using the static model!", "success")
            else:
                flash("✅ Dataset uploaded. (Static model not loaded, predictions unavailable)", "success")

            uploaded_df = df.copy()

        except Exception as e:
            flash(f"Error processing file for batch inference: {str(e)}", "error")
            return redirect(url_for("upload"))

        return redirect(url_for("upload"))

    return render_template("upload.html")


# ── SEARCH ─────────────────────────────────────────────────────────────────────
@app.route("/search")
def search():
    if not is_model_trained:
        flash("Please upload and train a model first.", "error")
        return redirect(url_for("upload"))
    return render_template("search.html")


@app.route("/search_customer")
def search_customer():
    """AJAX endpoint: return matching rows as JSON."""
    if not is_model_trained or uploaded_df is None:
        return jsonify({"results": []})

    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify({"results": []})

    df = uploaded_df.copy()

    # Build mask: match person_name OR person_id (case-insensitive, partial)
    mask = pd.Series([False] * len(df))
    for col in ["person_name", "name", "person_id", "id", "customer_id"]:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(q, na=False)

    results = df[mask].head(10)
    # Replace NaN with None for JSON serialisation
    records = results.where(pd.notnull(results), None).to_dict(orient="records")
    return jsonify({"results": records})


# ── PREDICT ────────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not is_model_trained:
        flash("Please upload and train a model first.", "error")
        return redirect(url_for("upload"))

    if request.method == "GET":
        return render_template("input.html")

    # ── POST: run prediction ──────────────────────────────────────────────────
    form = request.form.to_dict()

    # Collect metadata (not model features)
    person_name = form.get("person_name", "Applicant")
    person_id   = form.get("person_id", "")

    # Encode categoricals
    try:
        features = {
            "age":                        float(form.get("person_age", 0)),
            "annual_income":              float(form.get("person_income", 0)),
            "employment_years":           float(form.get("person_emp_length", 0)),
            "loan_amount":                float(form.get("loan_amnt", 0)),
            "debt_to_income_ratio":       float(form.get("loan_percent_income", 0)),
            "credit_score":               float(form.get("loan_int_rate", 650)) * 50, # Rough proxy if missing from UI
            "existing_loans":             1,
            "dependents":                 0,
            "late_payments_last_2yrs":    0,
            "savings_balance":            float(form.get("person_income", 0)) * 0.1,
            "loan_term_months":           36,
            "property_value":             0,
            "employment_type":            "Salaried" if form.get("person_home_ownership") == "MORTGAGE" else "Self-employed",
            "loan_purpose":               form.get("loan_intent", "Personal").title(),
        }

        # Apply get_dummies aligning to _features_list
        df_infer = pd.DataFrame([features])
        cat_cols = ['employment_type', 'loan_purpose']
        
        # Only encode those that exist in df_infer
        cat_cols = [c for c in cat_cols if c in df_infer.columns]
        df_encoded = pd.get_dummies(df_infer, columns=cat_cols)
        
        # Align with training features
        for col in _features_list:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[_features_list]

        # Scale
        X = df_encoded.values
        if _scaler is not None:
            X = _scaler.transform(X)

        # Predict using Random Forest
        if _model is not None:
            # predict_proba returns [[prob_default, prob_repay]]
            # We want prob_repay for consistency with the rest of the code
            prob_raw = _model.predict_proba(X)[0][1]
        else:
            # Demo fallback if model not available
            prob_raw = float(np.random.random())

        probability = float(prob_raw)
        
        # 3-Tier Risk Logic
        default_probability = 1.0 - probability
        if default_probability < 0.30:
            prediction = "Likely to Repay"
        elif default_probability <= 0.60:
            prediction = "Needs Review"
        else:
            prediction = "High Default Risk"

        # Store result in session for PDF download
        session["last_result"] = {
            "person_name": person_name,
            "person_id":   person_id,
            "prediction":  prediction,
            "probability": probability,
            "input_data":  form,
        }

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "error")
        return redirect(url_for("predict"))

    return render_template(
        "result.html",
        person_name=person_name,
        person_id=person_id,
        prediction=prediction,
        probability=probability,
        input_data=form,
    )


# ── PDF REPORT ─────────────────────────────────────────────────────────────────
@app.route("/download_report")
def download_report():
    result = session.get("last_result")
    if not result:
        flash("No prediction to export. Run a prediction first.", "error")
        return redirect(url_for("predict"))

    person_name = result["person_name"]
    person_id   = result["person_id"]
    prediction  = result["prediction"]
    probability = result["probability"]
    input_data  = result["input_data"]
    now         = datetime.now().strftime("%d %b %Y, %H:%M")
    repay       = prediction == "Likely to Repay" or prediction == "Needs Review"

    # ── ReportLab PDF ─────────────────────────────────────────────────────────
    if PDF_ENGINE == "reportlab":
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2.2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle("TitleS", parent=styles["Title"],
                                     fontSize=22, spaceAfter=4, textColor=colors.HexColor("#0d0f14"))
        sub_style   = ParagraphStyle("SubS",   parent=styles["Normal"],
                                     fontSize=11, textColor=colors.HexColor("#6b7280"))
        head_style  = ParagraphStyle("HeadS",  parent=styles["Heading2"],
                                     fontSize=12, spaceBefore=18, spaceAfter=6,
                                     textColor=colors.HexColor("#0d0f14"))
        body_style  = ParagraphStyle("BodyS",  parent=styles["Normal"],
                                     fontSize=10, leading=16,
                                     textColor=colors.HexColor("#374151"))

        story = []

        # ── Header ──
        story.append(Paragraph("LoanSight AI", title_style))
        story.append(Paragraph("Loan Repayment Prediction Report", sub_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#e5e4df")))
        story.append(Spacer(1, 0.4*cm))

        # ── Person Details ──
        story.append(Paragraph("Applicant Details", head_style))
        meta_data = [
            ["Person Name", person_name or "—"],
            ["Person ID",   person_id   or "—"],
            ["Report Date", now],
        ]
        meta_table = Table(meta_data, colWidths=[5*cm, 12*cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f5f4ef")),
            ("TEXTCOLOR",  (0,0), (0,-1), colors.HexColor("#6b7280")),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#fafaf8")]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#e5e4df")),
            ("PADDING",    (0,0), (-1,-1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Prediction Result ──
        story.append(Paragraph("Prediction Result", head_style))
        result_color = colors.HexColor("#10b981") if repay else colors.HexColor("#f43f5e")
        res_data = [
            ["Prediction",   prediction],
            ["Probability",  f"{probability*100:.1f}%"],
            ["Threshold",    "50.0%"],
            ["Risk Level",   "LOW" if repay else "HIGH"],
        ]
        res_table = Table(res_data, colWidths=[5*cm, 12*cm])
        res_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f5f4ef")),
            ("TEXTCOLOR",  (1,0), (1,0), result_color),
            ("FONTNAME",   (1,0), (1,0), "Helvetica-Bold"),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#fafaf8")]),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#e5e4df")),
            ("PADDING",    (0,0), (-1,-1), 8),
        ]))
        story.append(res_table)
        story.append(Spacer(1, 0.4*cm))

        # ── Input Features Table ──
        story.append(Paragraph("Input Features", head_style))
        feat_rows = [["Feature", "Value"]]
        for k, v in input_data.items():
            if k not in ("person_name", "person_id"):
                feat_rows.append([k.replace("_", " ").title(), str(v) if v else "—"])
        feat_table = Table(feat_rows, colWidths=[8*cm, 9*cm])
        feat_table.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#0d0f14")),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",     (0,0), (-1,-1), 9.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafaf8")]),
            ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#e5e4df")),
            ("PADDING",      (0,0), (-1,-1), 7),
        ]))
        story.append(feat_table)
        story.append(Spacer(1, 0.5*cm))

        # ── Summary ──
        story.append(Paragraph("Summary Explanation", head_style))
        if repay:
            summary = (f"The AI model predicts that {person_name or 'this applicant'} is LIKELY to repay "
                       f"the requested loan. With a repayment probability of {probability*100:.1f}%, "
                       f"the score exceeds the 50% decision threshold. "
                       f"{'High-confidence prediction — strong repayment signals detected.' if probability > 0.8 else 'Moderate confidence — standard due diligence is recommended.'}")
        else:
            summary = (f"The AI model predicts that {person_name or 'this applicant'} is UNLIKELY to repay "
                       f"the requested loan. With a repayment probability of {probability*100:.1f}%, "
                       f"the score falls below the 50% decision threshold. "
                       f"{'High-confidence prediction — significant risk factors detected.' if probability < 0.3 else 'Borderline prediction — manual review is advised.'}")
        story.append(Paragraph(summary, body_style))
        story.append(Spacer(1, 0.5*cm))

        # ── Footer line ──
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#e5e4df")))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "© 2026 LoanSight AI &nbsp;·&nbsp; Developed by: Your Name, Member 2, Member 3",
            ParagraphStyle("Foot", parent=styles["Normal"], fontSize=8,
                           textColor=colors.HexColor("#9ca3af"), alignment=1)
        ))

        doc.build(story)
        buf.seek(0)
        fname = f"LoanSight_Report_{(person_name or 'report').replace(' ','_')}.pdf"
        return send_file(buf, as_attachment=True,
                         download_name=fname, mimetype="application/pdf")

    # ── FPDF fallback ─────────────────────────────────────────────────────────
    elif PDF_ENGINE == "fpdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 12, "LoanSight AI — Prediction Report", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Generated: {now}", ln=True)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Applicant", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"Name: {person_name}   ID: {person_id}", ln=True)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Prediction Result", ln=True)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{prediction}  ({probability*100:.1f}%)", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Input Features", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in input_data.items():
            if k not in ("person_name", "person_id"):
                pdf.cell(0, 7, f"{k.replace('_',' ').title()}: {v or '—'}", ln=True)

        buf = io.BytesIO(pdf.output())
        fname = f"LoanSight_Report_{(person_name or 'report').replace(' ','_')}.pdf"
        return send_file(buf, as_attachment=True,
                         download_name=fname, mimetype="application/pdf")

    else:
        flash("PDF library not installed. Run: pip install reportlab", "error")
        return redirect(url_for("predict"))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
