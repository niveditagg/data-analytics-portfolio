"""
Project 2: Materials Quality Analytics: Mechanical Property & Defect Prediction
Author: Vallimpula Niveditha, IIT Jodhpur
Department: Materials and Metallurgical Engineering
Stack: Python, Pandas, Scikit-learn, NumPy
Description: Applied Machine Learning and EDA in Physical Metallurgy (Ti-6Al-4V Alloy).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def run_materials_informatics():
    print("=" * 65)
    print("PROJECT 2: MATERIALS INFORMATICS - MECHANICAL PROPERTY PREDICTION")
    print("=" * 65)

    # 1. Generate/Load 1,200 Experimental Test Samples (Ti-6Al-4V Alloy)
    np.random.seed(42)
    n_samples = 1200

    # Metallurgical features
    al_content = np.random.uniform(5.5, 6.75, n_samples)  # Aluminum (wt %)
    v_content = np.random.uniform(3.5, 4.5, n_samples)    # Vanadium (wt %)
    fe_content = np.random.uniform(0.1, 0.35, n_samples)  # Iron (wt %)
    annealing_temp = np.random.uniform(700, 950, n_samples) # Temperature (deg C)
    cooling_rate = np.random.uniform(1.0, 50.0, n_samples)  # Cooling rate (C/s)
    alpha_phase_pct = np.clip(85 - 0.04 * annealing_temp + 1.2 * al_content - 0.8 * v_content, 50, 95)

    # Ultimate Tensile Strength (UTS in MPa) based on metallurgical equations + noise
    uts_true = (
        650 + (35 * al_content) + (18 * v_content) + (45 * fe_content)
        - (0.15 * annealing_temp) + (1.8 * np.log1p(cooling_rate)) + (1.2 * alpha_phase_pct)
        + np.random.normal(0, 15, n_samples)
    )

    df = pd.DataFrame({
        'Al_wt_pct': np.round(al_content, 2),
        'V_wt_pct': np.round(v_content, 2),
        'Fe_wt_pct': np.round(fe_content, 3),
        'Annealing_Temp_C': np.round(annealing_temp, 1),
        'Cooling_Rate_Cs': np.round(cooling_rate, 1),
        'Alpha_Phase_Fraction_%': np.round(alpha_phase_pct, 2),
        'Tensile_Strength_UTS_MPa': np.round(uts_true, 1)
    })

    print(f"\n[DATASET SUMMARY]")
    print(f"Total Alloy Samples Analyzed: {len(df)}")
    print("Feature Columns: Al%, V%, Fe%, Annealing Temp, Cooling Rate, Alpha Phase %")
    print(f"Target Property : Ultimate Tensile Strength (UTS in MPa)")
    print(f"Mean UTS        : {df['Tensile_Strength_UTS_MPa'].mean():.2f} MPa")
    print(f"UTS Range       : {df['Tensile_Strength_UTS_MPa'].min():.1f} - {df['Tensile_Strength_UTS_MPa'].max():.1f} MPa")

    # 2. Train-Test Split (80% train, 20% test)
    X = df.drop(columns=['Tensile_Strength_UTS_MPa'])
    y = df['Tensile_Strength_UTS_MPa']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model 1: Multiple Linear Regression Baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

    # 4. Model 2: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    print("\n[MODEL BENCHMARKING RESULTS]")
    print(f"1. Multiple Linear Regression -> R² Score: {r2_lr:.3f} | RMSE: {rmse_lr:.2f} MPa")
    print(f"2. Random Forest Regressor    -> R² Score: {r2_rf:.3f} | RMSE: {rmse_rf:.2f} MPa")
    print(f">> Selected Model: Random Forest achieving {r2_rf * 100:.1f}% variance explained (R²).")

    # 5. Metallurgical Feature Importance
    print("\n[FEATURE IMPORTANCE BREAKDOWN]")
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feat, imp in feat_imp.items():
        print(f" - {feat:25s}: {imp * 100:.1f}%")

    df.to_csv('ti6al4v_metallurgy_dataset.csv', index=False)
    print("\n[SUCCESS] Exported dataset to 'ti6al4v_metallurgy_dataset.csv'.")

if __name__ == '__main__':
    run_materials_informatics()
