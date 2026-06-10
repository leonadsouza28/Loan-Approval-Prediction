"""
train.py
--------
Trains 4 ML models, compares them, and saves the best one.
Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

# Add parent so we can import preprocess.py
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data, preprocess

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR      = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
PLOT_DIR      = os.path.join(ARTIFACTS_DIR, 'model_plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)


def get_models():
    """Return a dict of (name → model) pairs to evaluate."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=42, n_jobs=-1
        ),
    }

    # XGBoost is optional — skip gracefully if not installed
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, use_label_encoder=False,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )
    except ImportError:
        print("⚠  XGBoost not installed — skipping. Run: pip install xgboost")

    return models


def evaluate_models(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                    cv: int = 5) -> pd.DataFrame:
    """Cross-validate every model and return a comparison DataFrame."""
    print("\n=== MODEL COMPARISON (5-fold Stratified CV) ===\n")
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        acc  = cross_val_score(model, X_train, y_train, cv=kf,
                                scoring='accuracy').mean()
        prec = cross_val_score(model, X_train, y_train, cv=kf,
                                scoring='precision').mean()
        rec  = cross_val_score(model, X_train, y_train, cv=kf,
                                scoring='recall').mean()
        f1   = cross_val_score(model, X_train, y_train, cv=kf,
                                scoring='f1').mean()
        roc  = cross_val_score(model, X_train, y_train, cv=kf,
                                scoring='roc_auc').mean()
        print(f"  {name:<25}  Acc={acc:.4f}  Prec={prec:.4f}"
              f"  Rec={rec:.4f}  F1={f1:.4f}  ROC-AUC={roc:.4f}")
        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                         'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc})

    df_results = pd.DataFrame(results).sort_values('Precision', ascending=False)
    return df_results


def plot_model_comparison(df_results: pd.DataFrame):
    """Bar chart comparing all models on all metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_melt = df_results.melt(id_vars='Model', value_vars=metrics,
                               var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model',
                palette='Set2', ax=ax, edgecolor='white')
    ax.set_title('Model Comparison – All Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right')
    path = os.path.join(PLOT_DIR, 'model_comparison.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ Model comparison chart → {path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    ax.set_title(f'Confusion Matrix – {model_name}', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    path = os.path.join(PLOT_DIR,
                        f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ Confusion matrix → {path}")


def plot_roc_curve(models_fitted: dict, X: np.ndarray, y: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models_fitted.items():
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probs)
            auc = roc_auc_score(y, probs)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves – All Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    path = os.path.join(PLOT_DIR, 'roc_curves.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ ROC curves → {path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, 'feature_importances_'):
        return
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.plot.barh(ax=ax, color='#3498db', edgecolor='white')
    ax.set_title(f'Feature Importance – {model_name}', fontweight='bold')
    ax.set_xlabel('Importance Score')
    path = os.path.join(PLOT_DIR,
                        f"feature_importance_{model_name.replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ Feature importance → {path}")


def train_and_save_best(X_train, y_train, X_test,
                        df_results: pd.DataFrame,
                        models: dict, feature_names: list):
    """Fit the best model on all training data and save it."""
    best_name = df_results.iloc[0]['Model']
    print(f"\n✨ Best model: {best_name} (highest ROC-AUC)")

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    # Full-train evaluation
    y_pred = best_model.predict(X_train)
    print("\nTrain-set performance of best model:")
    print(classification_report(y_train, y_pred,
                                 target_names=['Rejected', 'Approved']))

    # Plots
    plot_confusion_matrix(y_train, y_pred, best_name)
    plot_feature_importance(best_model, feature_names, best_name)

    # Save model
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    joblib.dump(best_name,  os.path.join(MODELS_DIR, 'best_model_name.pkl'))
    print(f"\n✔ Model saved → {model_path}")

    return best_model, best_name


def main():
    # 1. Load & preprocess
    print("Loading data …")
    train_df, test_df = load_data()

    print("Preprocessing …")
    X_train, y_train, X_test, encoders, imputers, scaler = \
        preprocess(train_df, test_df, fit=True)

    feature_names = list(X_train.columns) if hasattr(X_train, 'columns') \
                    else [f'f{i}' for i in range(X_train.shape[1])]
    X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_arr  = X_test.values  if hasattr(X_test,  'values') else X_test

    # 2. Get models
    models = get_models()

    # 3. Cross-validation comparison
    df_results = evaluate_models(models, X_train_arr, y_train.values)
    print("\n=== LEADERBOARD ===")
    print(df_results.to_string(index=False))
    df_results.to_csv(os.path.join(ARTIFACTS_DIR, 'model_comparison.csv'), index=False)

    # 4. Comparison chart
    plot_model_comparison(df_results)

    # 5. Fit all models (for ROC plot)
    models_fitted = {}
    for name, model in models.items():
        model.fit(X_train_arr, y_train.values)
        models_fitted[name] = model
    plot_roc_curve(models_fitted, X_train_arr, y_train.values)

    # 6. Save best model
    best_model, best_name = train_and_save_best(
        X_train_arr, y_train.values, X_test_arr,
        df_results, models, feature_names
    )

    print("\n✅  Training complete! All files saved.")


if __name__ == '__main__':
    main()
