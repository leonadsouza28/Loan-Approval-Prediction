"""
preprocess.py
-------------
Handles all data cleaning, encoding, and feature engineering for the
Loan Approval Prediction project.

Dataset columns:
  Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
  ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
  Credit_History, Property_Area, Loan_Status (target)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), '..', 'data')
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Column groups ──────────────────────────────────────────────────────
CATEGORICAL_COLS = ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Property_Area']
NUMERICAL_COLS   = ['ApplicantIncome', 'CoapplicantIncome',
                    'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
TARGET_COL       = 'Loan_Status'
DROP_COLS        = ['Loan_ID']   # not useful for prediction


def load_data(train_path=None, test_path=None):
    """Load raw CSVs from the data/ folder."""
    train_path = train_path or os.path.join(DATA_DIR, 'train_data.csv')
    test_path  = test_path  or os.path.join(DATA_DIR, 'test_data.csv')

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")
    return train, test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new meaningful features from existing ones.
    Called BEFORE imputation so that derived columns can be imputed too.
    """
    df = df.copy()

    # Total household income
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # Log-transform skewed income / loan columns (reduces outlier effect)
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome']:
        df[f'{col}_log'] = np.log1p(df[col])

    # EMI = LoanAmount / Loan_Amount_Term  (monthly installment proxy)
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']

    # Balance Income = income after EMI
    df['Balance_Income'] = df['TotalIncome'] - (df['EMI'] * 1000)

    return df


def preprocess(train: pd.DataFrame, test: pd.DataFrame,
               fit: bool = True,
               encoders: dict = None,
               imputers: dict = None,
               scaler: StandardScaler = None):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    train, test : DataFrames
    fit         : if True, fit encoders/imputers/scaler on train and save them.
                  if False, load saved artifacts and only transform.
    encoders    : dict of LabelEncoders (used when fit=False)
    imputers    : dict of SimpleImputers  (used when fit=False)
    scaler      : StandardScaler          (used when fit=False)

    Returns
    -------
    X_train, y_train, X_test, encoders, imputers, scaler
    """

    # ── 1. Drop ID column ──────────────────────────────────────────────
    train = train.drop(columns=DROP_COLS, errors='ignore')
    test  = test.drop(columns=DROP_COLS,  errors='ignore')

    # ── 2. Feature engineering ─────────────────────────────────────────
    train = engineer_features(train)
    test  = engineer_features(test)

    # ── 3. Separate target ─────────────────────────────────────────────
    y_train = None
    if TARGET_COL in train.columns:
        # Encode Y → 1, N → 0
        y_train = (train[TARGET_COL] == 'Y').astype(int)
        train   = train.drop(columns=[TARGET_COL])

    # ── 4. Identify all numeric & categorical cols after engineering ───
    num_cols = [c for c in train.columns
                if pd.api.types.is_numeric_dtype(train[c])]
    cat_cols = [c for c in train.columns
                if not pd.api.types.is_numeric_dtype(train[c])]

    # ── 5. Impute missing values ────────────────────────────────────────
    if fit:
        imputers = {}

        # Categorical → most frequent
        cat_imputer = SimpleImputer(strategy='most_frequent')
        train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
        test[cat_cols]  = cat_imputer.transform(test[cat_cols])
        imputers['cat'] = cat_imputer

        # Numerical → median
        num_imputer = SimpleImputer(strategy='median')
        train[num_cols] = num_imputer.fit_transform(train[num_cols])
        test[num_cols]  = num_imputer.transform(test[num_cols])
        imputers['num'] = num_imputer

        joblib.dump(imputers, os.path.join(ARTIFACTS_DIR, 'imputers.pkl'))
        print("✔ Imputers saved.")

    else:
        train[cat_cols] = imputers['cat'].transform(train[cat_cols])
        test[cat_cols]  = imputers['cat'].transform(test[cat_cols])
        train[num_cols] = imputers['num'].transform(train[num_cols])
        test[num_cols]  = imputers['num'].transform(test[num_cols])

    # ── 6. Label-encode categorical columns ────────────────────────────
    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
            test[col]  = le.transform(test[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, os.path.join(ARTIFACTS_DIR, 'encoders.pkl'))
        print("✔ Encoders saved.")

    else:
        for col in cat_cols:
            le = encoders[col]
            train[col] = le.transform(train[col].astype(str))
            test[col]  = le.transform(test[col].astype(str))

    # ── 7. Scale numerical columns ─────────────────────────────────────
    if fit:
        scaler = StandardScaler()
        train[num_cols] = scaler.fit_transform(train[num_cols])
        test[num_cols]  = scaler.transform(test[num_cols])
        joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
        print("✔ Scaler saved.")

    else:
        train[num_cols] = scaler.transform(train[num_cols])
        test[num_cols]  = scaler.transform(test[num_cols])

    print(f"\nFinal feature count : {train.shape[1]}")
    print(f"Features            : {list(train.columns)}")

    return train, y_train, test, encoders, imputers, scaler


def preprocess_single_input(input_dict: dict) -> np.ndarray:
    """
    Preprocess a single applicant's data (from Streamlit form) for prediction.
    Loads saved artifacts automatically.
    """
    encoders = joblib.load(os.path.join(ARTIFACTS_DIR, 'encoders.pkl'))
    imputers = joblib.load(os.path.join(ARTIFACTS_DIR, 'imputers.pkl'))
    scaler   = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))

    df = pd.DataFrame([input_dict])

    # Feature engineering
    df = engineer_features(df)

    # Identify column types
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    # Impute
    df[cat_cols] = imputers['cat'].transform(df[cat_cols])
    df[num_cols] = imputers['num'].transform(df[num_cols])

    # Encode
    for col in cat_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Scale
    df[num_cols] = scaler.transform(df[num_cols])

    return df.values


# ── Quick sanity check ─────────────────────────────────────────────────
if __name__ == '__main__':
    train_df, test_df = load_data()
    X_train, y_train, X_test, enc, imp, sc = preprocess(train_df, test_df, fit=True)
    print("\nClass distribution:")
    print(y_train.value_counts())
    print("\nPreprocessing complete ✔")
