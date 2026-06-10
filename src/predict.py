"""
predict.py
----------
Loads the saved best model and makes predictions.
Used by both the Streamlit app and batch inference scripts.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_single_input

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_model():
    """Load saved model and its name."""
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    name_path  = os.path.join(MODELS_DIR, 'best_model_name.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Please run `python src/train.py` first.")

    model = joblib.load(model_path)
    name  = joblib.load(name_path) if os.path.exists(name_path) else "Unknown"
    return model, name


def predict(input_dict: dict) -> dict:
    """
    Predict loan approval for a single applicant.

    Parameters
    ----------
    input_dict : dict with keys matching raw feature names:
        Gender, Married, Dependents, Education, Self_Employed,
        ApplicantIncome, CoapplicantIncome, LoanAmount,
        Loan_Amount_Term, Credit_History, Property_Area

    Returns
    -------
    dict with keys:
        prediction  : 'Approved' | 'Rejected'
        probability : float  (probability of approval, 0-1)
        model_used  : str
    """
    model, model_name = load_model()

    # Preprocess the raw input
    X = preprocess_single_input(input_dict)

    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None

    label = 'Approved' if pred == 1 else 'Rejected'
    return {
        'prediction' : label,
        'probability': round(float(prob), 4) if prob is not None else None,
        'model_used' : model_name,
    }


if __name__ == '__main__':
    # Quick test
    sample = {
        'Gender'           : 'Male',
        'Married'          : 'Yes',
        'Dependents'       : '0',
        'Education'        : 'Graduate',
        'Self_Employed'    : 'No',
        'ApplicantIncome'  : 5000,
        'CoapplicantIncome': 0,
        'LoanAmount'       : 120,
        'Loan_Amount_Term' : 360,
        'Credit_History'   : 1.0,
        'Property_Area'    : 'Urban',
    }
    result = predict(sample)
    print("Prediction :", result['prediction'])
    print("Probability:", result['probability'])
    print("Model used :", result['model_used'])
