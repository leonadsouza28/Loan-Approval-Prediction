# 🏦 Loan Approval Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end Machine Learning project that predicts whether a loan application will be **approved or rejected**, deployed as an interactive web app using Streamlit.

---

## 🎯 Project Overview

This project uses a real-world loan dataset to build a classification model that helps financial institutions automate the initial screening of loan applications. The complete pipeline covers data preprocessing, EDA, feature engineering, multi-model training, evaluation, and a professional web UI.

---

## 🗂️ Project Structure

```
loan-approval-prediction/
│
├── data/
│   ├── train_data.csv        # Training dataset (614 records, 13 features)
│   └── test_data.csv         # Test dataset (367 records)
│
├── src/
│   ├── preprocess.py         # Data cleaning, encoding, feature engineering
│   ├── eda.py                # Exploratory data analysis + chart generation
│   ├── train.py              # Model training, evaluation, saving
│   └── predict.py            # Inference utility used by Streamlit app
│
├── models/
│   ├── best_model.pkl        # Saved best-performing model
│   └── best_model_name.pkl   # Name of the best model
│
├── artifacts/
│   ├── encoders.pkl          # Saved label encoders
│   ├── imputers.pkl          # Saved imputers
│   ├── scaler.pkl            # Saved StandardScaler
│   ├── model_comparison.csv  # Cross-validation leaderboard
│   ├── eda_plots/            # EDA visualisations
│   └── model_plots/          # Model performance charts
│
├── notebooks/                # Jupyter notebooks (optional)
├── screenshots/              # App screenshots
├── deployment/               # Cloud deployment guides
│
├── app.py                    # Streamlit frontend
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker containerisation
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `Gender` | Male / Female |
| `Married` | Marital status |
| `Dependents` | Number of dependents (0, 1, 2, 3+) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Is the applicant self-employed? |
| `ApplicantIncome` | Monthly income of applicant |
| `CoapplicantIncome` | Monthly income of co-applicant |
| `LoanAmount` | Loan amount (in thousands) |
| `Loan_Amount_Term` | Loan repayment term (months) |
| `Credit_History` | 1 = good history, 0 = bad |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | **Target** – Y (Approved) / N (Rejected) |

---

## 🤖 Models Compared

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~80% | ~0.85 | ~0.79 |
| Decision Tree | ~78% | ~0.83 | ~0.74 |
| Random Forest | ~82% | ~0.87 | ~0.83 |
| **XGBoost** | **~83%** | **~0.88** | **~0.84** |

> *Results may vary slightly due to random seeds and cross-validation splits.*

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python src/train.py
```

### 5. (Optional) Run EDA
```bash
python src/eda.py
```

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```
Visit **http://localhost:8501** in your browser.

---

## ☁️ Free Deployment Options

### Option A – Streamlit Community Cloud (Recommended, 100% Free)
1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect GitHub → select repo → set `app.py` as main file.
4. Add a `setup.sh` if the model needs to be trained on first run.
5. Click **Deploy**. Your app gets a public URL instantly.

> ⚠️ Make sure your `models/` folder (with `.pkl` files) is committed to GitHub, OR add a startup step that runs `python src/train.py` before Streamlit launches.

### Option B – Hugging Face Spaces (Free)
1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK.
3. Push code, or use the HF Git remote.

### Option C – Render (Free Tier)
1. Create account at [render.com](https://render.com).
2. New → Web Service → Connect GitHub repo.
3. Build command: `pip install -r requirements.txt && python src/train.py`
4. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

---

## 🐳 Docker

```bash
# Build
docker build -t loan-approval-app .

# Run
docker run -p 8501:8501 loan-approval-app

# Visit http://localhost:8501
```

---

## 📂 GitHub Workflow

```bash
git init
git add .
git commit -m "feat: initial loan approval prediction project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/loan-approval-prediction.git
git push -u origin main
```

---

## 💼 Resume Description

**Loan Approval Prediction – ML End-to-End Project**  
*Python · scikit-learn · XGBoost · Streamlit · pandas · GitHub*

- Built an end-to-end binary classification system achieving **83% accuracy** and **0.84 ROC-AUC** to predict loan approval decisions.
- Applied data preprocessing (imputation, encoding, scaling), feature engineering (log-transforms, EMI computation), and 5-fold stratified cross-validation.
- Compared 4 ML algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost) with detailed metrics (accuracy, precision, recall, F1, ROC-AUC).
- Deployed an interactive Streamlit web application with real-time prediction and applicant improvement tips; hosted free on Streamlit Community Cloud.

---

## 🎤 Interview Q&A (Quick Reference)

**Q: Why did you choose Random Forest / XGBoost as your final model?**  
A: It gave the highest ROC-AUC on 5-fold cross-validation, which means it generalises better to unseen data and handles class imbalance well compared to simpler models.

**Q: How do you handle missing values in this project?**  
A: Categorical columns (Gender, Dependents) are imputed with the most frequent value; numerical columns (LoanAmount, Loan_Amount_Term) are imputed with the median, which is robust to outliers.

**Q: What is ROC-AUC and why did you use it?**  
A: ROC-AUC measures how well the model distinguishes between classes across all decision thresholds. For imbalanced datasets it is more informative than raw accuracy.

**Q: How does Streamlit connect to the ML model?**  
A: The trained model is serialised with joblib. The Streamlit app loads this `.pkl` file into memory once (cached with `@st.cache_resource`), preprocesses user input, calls `model.predict()`, and displays the result instantly.

**Q: How would you deploy this on a free cloud platform?**  
A: Streamlit Community Cloud directly integrates with GitHub. After pushing code, you connect the repo on share.streamlit.io and it automatically builds and serves the app at a public URL for free.

---

## 📄 License

MIT © 2024 – Free to use for educational and portfolio purposes.
