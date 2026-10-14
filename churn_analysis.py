"""
Project 3: Telecom Customer Churn Exploratory Data Analysis & Predictive Modeling
Author: Vallimpula Niveditha, IIT Jodhpur
Stack: Python (Pandas, Scikit-learn, NumPy)
Description: End-to-end churn analysis, customer behavior EDA, and ML classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_churn_analysis():
    print("=" * 65)
    print("PROJECT 3: TELECOM CUSTOMER CHURN & BEHAVIORAL ANALYTICS")
    print("=" * 65)

    # 1. Simulate/Load Telecom Dataset (7,043 customer records)
    np.random.seed(42)
    n_records = 7043

    contracts = ['Month-to-month', 'One year', 'Two year']
    tech_supports = ['No', 'Yes', 'No internet service']
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']

    tenure = np.random.randint(1, 72, n_records)
    monthly_charges = np.round(np.random.uniform(20.0, 118.0, n_records), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 10, n_records), 2)
    total_charges = np.clip(total_charges, 20.0, None)
    contract_choice = np.random.choice(contracts, n_records, p=[0.55, 0.25, 0.20])

    # Churn probability higher for month-to-month and higher charges
    churn_prob = (
        0.15 
        + 0.35 * (contract_choice == 'Month-to-month') 
        - 0.20 * (tenure > 36) 
        + 0.15 * (monthly_charges > 70)
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = (np.random.rand(n_records) < churn_prob).astype(int)

    df = pd.DataFrame({
        'CustomerID': [f'TEL-{10000 + i}' for i in range(n_records)],
        'Tenure_Months': tenure,
        'Contract': contract_choice,
        'TechSupport': np.random.choice(tech_supports, n_records, p=[0.5, 0.3, 0.2]),
        'PaymentMethod': np.random.choice(payment_methods, n_records),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })

    print(f"\n[DATASET OVERVIEW]")
    print(f"Total Customer Records : {len(df):,}")
    churn_rate = (df['Churn'].sum() / len(df)) * 100
    print(f"Overall Churn Rate     : {churn_rate:.2f}%")

    # 2. Key Bivariate EDA Findings
    print("\n[CHURN RATE BY CONTRACT TYPE]")
    contract_churn = df.groupby('Contract')['Churn'].agg(
        Total_Customers='count',
        Churn_Rate=lambda x: f"{x.mean() * 100:.1f}%"
    ).reset_index()
    print(contract_churn.to_string(index=False))

    # 3. Data Preprocessing & One-Hot Encoding
    categorical_cols = ['Contract', 'TechSupport', 'PaymentMethod']
    df_encoded = pd.get_dummies(df.drop(columns=['CustomerID']), columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Models
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    print("\n[MODEL EVALUATION]")
    print(f"1. Logistic Regression Accuracy : {acc_lr * 100:.2f}%")
    print(f"2. Decision Tree Classifier Acc : {acc_dt * 100:.2f}%")

    # Feature Importance from Decision Tree
    print("\n[TOP CHURN RISK DRIVERS]")
    dt_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feat, imp in dt_importances.head(5).items():
        print(f" - {feat:30s}: {imp * 100:.1f}%")

    df.to_csv('telecom_churn_data.csv', index=False)
    print("\n[SUCCESS] Exported dataset to 'telecom_churn_data.csv'.")

if __name__ == '__main__':
    run_churn_analysis()
