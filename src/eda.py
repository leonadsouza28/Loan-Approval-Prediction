"""
eda.py
------
Exploratory Data Analysis for the Loan Approval Prediction project.
Run this script to generate all charts saved inside artifacts/eda_plots/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
PLOT_DIR  = os.path.join(BASE_DIR, 'artifacts', 'eda_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120


def save(fig, name):
    path = os.path.join(PLOT_DIR, f'{name}.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ Saved → {path}")


def run_eda(df: pd.DataFrame):
    print("\n=== EXPLORATORY DATA ANALYSIS ===\n")

    # ── 1. Basic info ──────────────────────────────────────────────────
    print("Shape          :", df.shape)
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nBasic stats:\n", df.describe())

    target = 'Loan_Status'

    # ── 2. Target distribution ─────────────────────────────────────────
    print("\n[1] Target Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vc = df[target].value_counts()
    vc.plot.bar(ax=axes[0], color=['#2ecc71', '#e74c3c'], edgecolor='white')
    axes[0].set_title('Loan Approval Count')
    axes[0].set_xlabel('Loan Status (Y = Approved)')
    axes[0].set_ylabel('Count')
    vc.plot.pie(ax=axes[1], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                startangle=90, labels=['Approved', 'Rejected'])
    axes[1].set_ylabel('')
    axes[1].set_title('Approval Rate')
    fig.suptitle('Target Variable Distribution', fontsize=14, fontweight='bold')
    save(fig, '01_target_distribution')

    # ── 3. Categorical features vs target ─────────────────────────────
    print("[2] Categorical Features vs Target")
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
                'Self_Employed', 'Property_Area', 'Credit_History']
    n = len(cat_cols)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        ct = pd.crosstab(df[col], df[target])
        ct.plot.bar(ax=axes[i], stacked=True,
                    color=['#e74c3c', '#2ecc71'], edgecolor='white')
        axes[i].set_title(f'{col} vs Loan Status')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', rotation=30)
        axes[i].legend(['Rejected', 'Approved'])
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Categorical Features vs Loan Status', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save(fig, '02_categorical_vs_target')

    # ── 4. Numerical distributions ─────────────────────────────────────
    print("[3] Numerical Distributions")
    num_cols = ['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue=target, kde=True,
                     ax=axes[i], palette=['#e74c3c', '#2ecc71'])
        axes[i].set_title(f'Distribution of {col}')
    fig.suptitle('Numerical Feature Distributions', fontsize=15, fontweight='bold')
    plt.tight_layout()
    save(fig, '03_numerical_distributions')

    # ── 5. Box plots (outlier detection) ──────────────────────────────
    print("[4] Box Plots")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']):
        sns.boxplot(data=df, x=target, y=col, ax=axes[i],
                    palette=['#e74c3c', '#2ecc71'])
        axes[i].set_title(f'{col} by Loan Status')
    fig.suptitle('Outlier Detection via Box Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save(fig, '04_boxplots')

    # ── 6. Correlation heatmap ─────────────────────────────────────────
    print("[5] Correlation Heatmap")
    num_df = df[['ApplicantIncome', 'CoapplicantIncome',
                 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']].copy()
    num_df['Loan_Status_bin'] = (df[target] == 'Y').astype(int)
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                mask=mask, ax=ax, linewidths=0.5,
                vmin=-1, vmax=1, square=True)
    ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    save(fig, '05_correlation_heatmap')

    # ── 7. Credit History impact ───────────────────────────────────────
    print("[6] Credit History Impact")
    fig, ax = plt.subplots(figsize=(8, 5))
    ct = pd.crosstab(df['Credit_History'], df[target], normalize='index') * 100
    ct.plot.bar(ax=ax, color=['#e74c3c', '#2ecc71'], edgecolor='white')
    ax.set_title('Approval Rate by Credit History', fontsize=13, fontweight='bold')
    ax.set_xlabel('Credit History (1 = Good, 0 = Bad)')
    ax.set_ylabel('Percentage (%)')
    ax.legend(['Rejected', 'Approved'])
    ax.tick_params(axis='x', rotation=0)
    save(fig, '06_credit_history_impact')

    # ── 8. Income vs Loan Amount scatter ───────────────────────────────
    print("[7] Income vs Loan Amount")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = df[target].map({'Y': '#2ecc71', 'N': '#e74c3c'})
    ax.scatter(df['ApplicantIncome'], df['LoanAmount'],
               c=colors, alpha=0.6, edgecolors='white', linewidth=0.4, s=60)
    ax.set_xlabel('Applicant Income')
    ax.set_ylabel('Loan Amount')
    ax.set_title('Applicant Income vs Loan Amount', fontsize=13, fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Approved'),
                       Patch(facecolor='#e74c3c', label='Rejected')]
    ax.legend(handles=legend_elements)
    save(fig, '07_income_vs_loanamount')

    print("\n✔ All EDA plots saved to:", PLOT_DIR)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    run_eda(df)
