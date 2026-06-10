"""
app.py  –  Streamlit frontend for Loan Approval Prediction
-----------------------------------------------------------
Run:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup so we can import src/ modules ───────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-approved {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .result-rejected {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3498db;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2c3e50, #3498db);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.7rem;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3498db, #2c3e50);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: load model once ─────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, 'best_model.pkl')
    name_path  = os.path.join(models_dir, 'best_model_name.pkl')

    if not os.path.exists(model_path):
        return None, None

    model = joblib.load(model_path)
    name  = joblib.load(name_path) if os.path.exists(name_path) else "Best Model"
    return model, name


@st.cache_resource
def load_artifacts():
    artifacts_dir = os.path.join(os.path.dirname(__file__), 'artifacts')
    try:
        encoders = joblib.load(os.path.join(artifacts_dir, 'encoders.pkl'))
        imputers = joblib.load(os.path.join(artifacts_dir, 'imputers.pkl'))
        scaler   = joblib.load(os.path.join(artifacts_dir, 'scaler.pkl'))
        return encoders, imputers, scaler
    except Exception:
        return None, None, None


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank.png", width=80)
    st.markdown("## 🏦 About This App")
    st.markdown("""
    This app predicts whether a loan application will be **approved** or
    **rejected** using a machine learning model trained on historical data.

    ---
    **Models Evaluated:**
    - ✅ Logistic Regression
    - ✅ Decision Tree
    - ✅ Random Forest
    - ✅ XGBoost

    ---
    **Dataset:** 614 loan applications from a finance company.

    ---
    **Stack:**
    `Python · scikit-learn · Streamlit · pandas · joblib`

    ---
    💡 *Fill in the applicant details on the right and click **Predict**.*
    """)

    st.markdown("---")
    st.markdown("**Made by a Data Science Student**")
    st.markdown("[GitHub](https://github.com) · [LinkedIn](https://linkedin.com)")


# ── Main title ─────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏦 Loan Approval Predictor</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict your loan eligibility in seconds using Machine Learning</div>',
            unsafe_allow_html=True)

# ── Tab layout ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 EDA Insights", "ℹ️ Model Info"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION FORM
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Fill Applicant Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("No. of Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    with col2:
        st.markdown("#### 💰 Financial Info")
        applicant_income = st.number_input(
            "Applicant Monthly Income (₹)", min_value=0, max_value=1_000_000,
            value=5000, step=500,
            help="Enter your monthly income in Indian Rupees.")
        coapplicant_income = st.number_input(
            "Co-Applicant Monthly Income (₹)", min_value=0, max_value=500_000,
            value=0, step=500)
        loan_amount = st.number_input(
            "Loan Amount (in thousands ₹)", min_value=1, max_value=1000,
            value=120, step=10,
            help="Enter the loan amount in thousands. E.g., 150 means ₹1,50,000.")
        loan_term = st.selectbox(
            "Loan Amount Term (months)", [360, 180, 120, 60, 36, 24, 12, 480, 300, 240, 84],
            index=0)

    with col3:
        st.markdown("#### 🏠 Other Details")
        credit_history = st.selectbox(
            "Credit History",
            options=[1.0, 0.0],
            format_func=lambda x: "Good (meets guidelines)" if x == 1.0 else "Bad (does not meet guidelines)",
            help="Have you repaid your previous debts on time?")
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"])

        # Summary card
        st.markdown("---")
        st.markdown("#### 📋 Summary")
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | Gender | {gender} |
        | Education | {education} |
        | Income | ₹{applicant_income:,} |
        | Loan | ₹{loan_amount*1000:,} |
        | Area | {property_area} |
        """)

    st.markdown("---")

    # Predict button
    predict_btn = st.button("🔮 Predict Loan Approval", use_container_width=True)

    if predict_btn:
        model, model_name = load_model_cached()

        if model is None:
            st.error("⚠️ Model not found! Please run `python src/train.py` first "
                     "to train and save the model.")
        else:
            input_dict = {
                'Gender'           : gender,
                'Married'          : married,
                'Dependents'       : dependents,
                'Education'        : education,
                'Self_Employed'    : self_employed,
                'ApplicantIncome'  : applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount'       : float(loan_amount),
                'Loan_Amount_Term' : float(loan_term),
                'Credit_History'   : float(credit_history),
                'Property_Area'    : property_area,
            }

            try:
                from predict import predict
                result = predict(input_dict)

                st.markdown("---")
                res_col1, res_col2 = st.columns([2, 1])

                with res_col1:
                    if result['prediction'] == 'Approved':
                        st.markdown(
                            f'<div class="result-approved">'
                            f'✅ LOAN APPROVED<br>'
                            f'<span style="font-size:1rem;font-weight:400;">'
                            f'Congratulations! Your loan is likely to be approved.</span>'
                            f'</div>', unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(
                            f'<div class="result-rejected">'
                            f'❌ LOAN REJECTED<br>'
                            f'<span style="font-size:1rem;font-weight:400;">'
                            f'Unfortunately, this application may not be approved.</span>'
                            f'</div>', unsafe_allow_html=True)

                with res_col2:
                    if result['probability'] is not None:
                        prob_pct = result['probability'] * 100
                        st.metric("Approval Probability",
                                  f"{prob_pct:.1f}%",
                                  delta=f"{prob_pct - 50:.1f}% vs 50% baseline")
                        # Simple gauge-like bar
                        st.progress(result['probability'])
                    st.caption(f"🤖 Model: {model_name}")

                # Tips if rejected
                if result['prediction'] == 'Rejected':
                    st.markdown("---")
                    st.markdown("#### 💡 How to improve your chances:")
                    tips = []
                    if credit_history == 0.0:
                        tips.append("• **Improve Credit History** – Repay existing debts on time.")
                    if applicant_income < 3000:
                        tips.append("• **Increase Income** – Consider additional income sources.")
                    if coapplicant_income == 0:
                        tips.append("• **Add a Co-Applicant** – A co-applicant improves eligibility.")
                    if loan_amount > 200:
                        tips.append("• **Reduce Loan Amount** – Apply for a smaller loan first.")
                    if not tips:
                        tips = ["• Review all financial commitments and reapply after 6 months."]
                    for tip in tips:
                        st.markdown(tip)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure you've run `python src/train.py` and all artifact files exist.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 – EDA INSIGHTS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Dataset Insights")

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'train_data.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        # Key stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Applicants", len(df))
        m2.metric("Approved",
                  int((df['Loan_Status'] == 'Y').sum()),
                  f"{(df['Loan_Status']=='Y').mean()*100:.1f}%")
        m3.metric("Avg Income",
                  f"₹{int(df['ApplicantIncome'].mean()):,}")
        m4.metric("Avg Loan",
                  f"₹{int(df['LoanAmount'].mean()*1000):,}")

        st.markdown("---")

        # Show plots if they exist
        plot_dir = os.path.join(os.path.dirname(__file__),
                                'artifacts', 'eda_plots')
        plot_map = {
            '01_target_distribution.png'  : '📊 Loan Approval Distribution',
            '02_categorical_vs_target.png': '🗂️ Categorical Features vs Approval',
            '03_numerical_distributions.png': '📈 Numerical Feature Distributions',
            '05_correlation_heatmap.png'  : '🔥 Correlation Heatmap',
            '06_credit_history_impact.png': '💳 Credit History Impact',
            '07_income_vs_loanamount.png' : '💰 Income vs Loan Amount',
        }

        for filename, title in plot_map.items():
            path = os.path.join(plot_dir, filename)
            if os.path.exists(path):
                st.markdown(f"#### {title}")
                st.image(path, use_column_width=True)
                st.markdown("---")
            else:
                st.info(f"Run `python src/eda.py` to generate: {filename}")

        # Raw data preview
        with st.expander("🔍 Preview Raw Dataset"):
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Showing 20 of {len(df)} rows.")
    else:
        st.warning("Training data not found at data/train_data.csv")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL INFO
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ℹ️ Model Information")

    model, model_name = load_model_cached()

    if model is not None:
        st.success(f"✅ Active Model: **{model_name}**")
    else:
        st.warning("No model loaded. Run `python src/train.py` first.")

    # Model comparison CSV
    comp_path = os.path.join(os.path.dirname(__file__),
                             'artifacts', 'model_comparison.csv')
    if os.path.exists(comp_path):
        st.markdown("#### 🏆 Model Comparison Leaderboard")
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df.style.highlight_max(
            subset=['Accuracy', 'F1-Score', 'ROC-AUC'], color='#d4edda'),
            use_container_width=True)

    # Model comparison chart
    plot_dir = os.path.join(os.path.dirname(__file__),
                            'artifacts', 'model_plots')
    comp_img = os.path.join(plot_dir, 'model_comparison.png')
    roc_img  = os.path.join(plot_dir, 'roc_curves.png')
    if os.path.exists(comp_img):
        st.markdown("#### 📊 Visual Comparison")
        st.image(comp_img, use_column_width=True)
    if os.path.exists(roc_img):
        st.markdown("#### 📈 ROC Curves")
        st.image(roc_img, use_column_width=True)

    st.markdown("---")
    st.markdown("""
    #### 🛠️ Tech Stack
    | Component | Technology |
    |---|---|
    | Language | Python 3.10+ |
    | ML Library | scikit-learn |
    | Boosting | XGBoost |
    | Frontend | Streamlit |
    | Data | pandas, numpy |
    | Visualization | matplotlib, seaborn |
    | Model Saving | joblib |
    | Deployment | Streamlit Cloud / Render / Hugging Face |
    """)
