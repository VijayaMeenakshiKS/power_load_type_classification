"""
Power System Load Type Classification - End-to-End Pipeline
=============================================================
This script executes the complete machine learning pipeline:
1. Intelligent data loading & schema normalization
2. Physics-informed & cyclical feature engineering
3. Strict chronological train/test split (last month as holdout test set)
4. Multi-model benchmarking (Baseline, LogReg, Random Forest, LightGBM, XGBoost)
5. Comprehensive metrics evaluation (Accuracy, Precision, Recall, Macro F1, Weighted F1)
6. Confusion matrix & feature importance export
7. Best model serialization for production / Streamlit serving
"""

import os
import sys
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Optional gradient boosting libraries with graceful fallback
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ==========================================
# 1. DATA INGESTION & DISCOVERY
# ==========================================
def is_valid_power_dataset(file_path: Path) -> bool:
    """Check if file actually contains power load columns."""
    try:
        if file_path.suffix.lower() in [".xlsx", ".xls"]:
            df_head = pd.read_excel(file_path, nrows=5)
        elif file_path.suffix.lower() == ".csv":
            try:
                df_head = pd.read_csv(file_path, nrows=5)
            except Exception:
                df_head = pd.read_csv(file_path, nrows=5, sep=None, engine='python')
        else:
            return False

        cols = [c.lower() for c in df_head.columns]
        # Look for essential power features
        has_power_signal = any(
            any(k in c for k in ['load', 'usage', 'kwh', 'nsm', 'lagging', 'reactive', 'leading'])
            for c in cols
        )
        return has_power_signal
    except Exception:
        return False


def find_dataset() -> Path:
    """Find the dataset in the project or use POWER_LOAD_DATA explicitly."""
    valid_exts = {".csv", ".xlsx", ".xls"}
    project_root = Path(__file__).resolve().parent
    configured = os.environ.get("POWER_LOAD_DATA")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        project_root / "data" / "data.csv",
        project_root / "data" / "Steel_industry_data.csv",
        project_root / "data",
        project_root / "data.csv",
        Path.cwd() / "data" / "data.csv",
        Path.cwd() / "data.csv",
    ])

    for candidate in candidates:
        files = ([candidate] if candidate.is_file() else
                 sorted(path for path in candidate.glob("*") if path.is_file()) if candidate.is_dir() else [])
        for file_path in files:
            if file_path.suffix.lower() in valid_exts and is_valid_power_dataset(file_path):
                print(f"[INFO] Found verified dataset file at: {file_path}")
                return file_path

    raise FileNotFoundError(
        "Power load dataset not found. Put it at data/data.csv in the project, "
        "or set POWER_LOAD_DATA to its full CSV/XLSX path."
    )


def load_and_normalize_data(file_path: Path) -> pd.DataFrame:
    """Loads CSV/Excel and standardizes column names across dataset variants."""
    print(f"\n--- Loading Data from: {file_path} ---")
    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=None, engine='python')

    print(f"Initial shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Flexible column mapping
    col_mapping = {}
    for col in df.columns:
        norm = col.strip().lower().replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
        if 'date' in norm:
            col_mapping[col] = 'Date'
        elif 'usage' in norm or 'kwh' in norm:
            col_mapping[col] = 'Usage_kWh'
        elif 'lagging' in norm and 'reactive' in norm:
            col_mapping[col] = 'Lagging_Current_Reactive_Power_kVarh'
        elif 'leading' in norm and 'reactive' in norm:
            col_mapping[col] = 'Leading_Current_Reactive_Power_kVarh'
        elif 'co2' in norm:
            col_mapping[col] = 'CO2'
        elif 'nsm' in norm:
            col_mapping[col] = 'NSM'
        elif 'load' in norm and 'type' in norm:
            col_mapping[col] = 'Load_Type'

    df = df.rename(columns=col_mapping)
    print(f"Normalized columns: {list(df.columns)}")

    required = ['Date', 'Usage_kWh', 'Lagging_Current_Reactive_Power_kVarh',
                'Leading_Current_Reactive_Power_kVarh', 'CO2', 'NSM', 'Load_Type']
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {list(df.columns)}")

    # Parse Date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce')
    df = df.dropna(subset=['Date']).copy()
    numeric_inputs = ['Usage_kWh', 'Lagging_Current_Reactive_Power_kVarh',
                      'Leading_Current_Reactive_Power_kVarh', 'CO2', 'NSM']
    df[numeric_inputs] = df[numeric_inputs].apply(pd.to_numeric, errors='coerce')
    df[numeric_inputs] = df[numeric_inputs].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=numeric_inputs + ['Load_Type'])
    print(f"Dropped {before - len(df):,} rows with missing required sensor readings.")
    df = df.sort_values('Date').reset_index(drop=True)

    # Standardize Load_Type labels
    df['Load_Type'] = (
        df['Load_Type']
        .astype(str)
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
    )
    print(f"Target class distribution:\n{df['Load_Type'].value_counts()}\n")

    return df


# ==========================================
# 2. FEATURE ENGINEERING (PHYSICS & CYCLICAL)
# ==========================================
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies electrical power triangle physics and cyclical time encodings.
    """
    df = data.copy()

    # --- A. Temporal & Cyclical Features ---
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Cyclical encoding for Seconds from Midnight (NSM: 0 to 86400)
    # Ensures 23:59:59 is mathematically adjacent to 00:00:00
    df['sin_NSM'] = np.sin(2 * np.pi * df['NSM'] / 86400.0)
    df['cos_NSM'] = np.cos(2 * np.pi * df['NSM'] / 86400.0)

    # Cyclical encoding for Hour of day
    df['sin_Hour'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['cos_Hour'] = np.cos(2 * np.pi * df['Hour'] / 24.0)

    # --- B. Electrical Power Triangle Features ---
    # Net reactive power: inductive lagging minus capacitive leading
    lag_q = df['Lagging_Current_Reactive_Power_kVarh']
    lead_q = df['Leading_Current_Reactive_Power_kVarh']
    p_active = df['Usage_kWh']

    df['Net_Reactive_Power_kVarh'] = lag_q - lead_q
    df['Total_Reactive_Power_kVarh'] = lag_q + lead_q

    # Apparent Power S = sqrt(P^2 + Q^2)
    df['Apparent_Power_kVAh'] = np.sqrt(p_active**2 + df['Net_Reactive_Power_kVarh']**2)

    # Operating Power Factor PF = P / S
    df['Operating_Power_Factor'] = p_active / (df['Apparent_Power_kVAh'] + 1e-6)
    # Clip between 0 and 1 for physical realism
    df['Operating_Power_Factor'] = df['Operating_Power_Factor'].clip(0.0, 1.0)

    # Reactive-to-Active Ratio (tan phi)
    df['Reactive_Active_Ratio'] = df['Net_Reactive_Power_kVarh'] / (p_active + 1e-6)

    # Carbon Intensity Ratio
    if 'CO2' in df.columns:
        df['Carbon_Intensity'] = df['CO2'] / (p_active + 1e-6)

    # --- C. Short-term Trend & Dynamics ---
    df['Usage_Lag_1'] = df['Usage_kWh'].shift(1).fillna(0.0)
    df['Usage_Delta_1'] = df['Usage_kWh'] - df['Usage_Lag_1']
    df['Usage_Rolling_Mean_4'] = df['Usage_kWh'].rolling(window=4, min_periods=1).mean()

    return df


# ==========================================
# 3. CHRONOLOGICAL VALIDATION SPLIT
# ==========================================
def chronological_split(df: pd.DataFrame):
    """
    Holds out strictly the last month of data as the test set.
    """
    df['YearMonth'] = df['Date'].dt.to_period('M')
    last_month = df['YearMonth'].max()
    print(f"--- Chronological Train/Test Split ---")
    print(f"Held-out Final Month (Test Set): {last_month}")

    train_df = df[df['YearMonth'] < last_month].copy()
    test_df = df[df['YearMonth'] == last_month].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("A last-month holdout requires at least two distinct calendar months of dated data.")

    print(f"Train period: {train_df['Date'].min()} -> {train_df['Date'].max()} ({len(train_df)} samples)")
    print(f"Test period:  {test_df['Date'].min()} -> {test_df['Date'].max()} ({len(test_df)} samples)")

    return train_df, test_df


# ==========================================
# 4. BENCHMARK SUITE & EVALUATION
# ==========================================
def run_pipeline():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    data_file = find_dataset()
    df_raw = load_and_normalize_data(data_file)
    df_feat = engineer_features(df_raw)

    train_df, test_df = chronological_split(df_feat)

    feature_cols = [
        'Usage_kWh',
        'Lagging_Current_Reactive_Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'CO2',
        'NSM',
        'Hour',
        'DayOfWeek',
        'Is_Weekend',
        'sin_NSM',
        'cos_NSM',
        'sin_Hour',
        'cos_Hour',
        'Net_Reactive_Power_kVarh',
        'Total_Reactive_Power_kVarh',
        'Apparent_Power_kVAh',
        'Operating_Power_Factor',
        'Reactive_Active_Ratio',
        'Carbon_Intensity',
        'Usage_Lag_1',
        'Usage_Delta_1',
        'Usage_Rolling_Mean_4'
    ]
    # Filter features that exist
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"\nFinal feature set ({len(feature_cols)} features):\n{feature_cols}\n")

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df['Load_Type'])
    y_test = le.transform(test_df['Load_Type'])
    class_names = list(le.classes_)
    print(f"Class mapping: {dict(enumerate(class_names))}")

    # Standard scale for linear model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models dictionary
    models = {
        "Dummy Baseline": (DummyClassifier(strategy="most_frequent"), False),
        "Logistic Regression": (LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1), False),
    }

    if HAS_LGBM:
        models["LightGBM"] = (
            lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
            False
        )

    if HAS_XGB:
        models["XGBoost"] = (
            xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='mlogloss'),
            False
        )

    results = []
    trained_models = {}

    print("\n--- Training & Benchmarking Models ---")
    for name, (model, requires_scale) in models.items():
        print(f"Training {name}...")
        X_tr = X_train_scaled if requires_scale else X_train
        X_te = X_test_scaled if requires_scale else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        trained_models[name] = model

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro Precision": prec_macro,
            "Macro Recall": rec_macro,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted
        })

    results_df = pd.DataFrame(results).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    print("\n================ MODEL BENCHMARK TABLE ================")
    print(results_df.to_string(index=False))
    results_df.to_csv("outputs/model_benchmark_results.csv", index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n[WINNER] Selected Best Model: {best_model_name} (Macro F1 = {results_df.iloc[0]['Macro F1']:.4f})")

    # Final Detailed Diagnostics on Best Model
    X_te_best = X_test_scaled if models[best_model_name][1] else X_test
    y_pred_best = best_model.predict(X_te_best)

    print("\n============== DETAILED CLASSIFICATION REPORT ==============")
    report_str = classification_report(y_test, y_pred_best, target_names=class_names, digits=4)
    print(report_str)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report_str)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred_best)
    cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(f"Normalized Confusion Matrix - {best_model_name}\n(Unseen Test Month)", fontsize=14, pad=15)
    plt.xlabel("Predicted Load Type", fontsize=12)
    plt.ylabel("True Load Type", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300)
    plt.close()
    print("[INFO] Saved confusion matrix to outputs/confusion_matrix.png")

    # Feature Importance Plot (if tree-based)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values[:12], y=importances.index[:12], palette="viridis")
        plt.title(f"Top 12 Feature Importances - {best_model_name}", fontsize=14)
        plt.xlabel("Relative Importance", fontsize=12)
        plt.tight_layout()
        plt.savefig("outputs/feature_importance.png", dpi=300)
        plt.close()
        print("[INFO] Saved feature importance to outputs/feature_importance.png")

    # Serialize Best Model Artifacts for Streamlit / Deployment
    artifact_package = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler if models[best_model_name][1] else None,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "classes": class_names,
        "benchmark_results": results_df.to_dict(orient="records")
    }
    model_save_path = "models/best_load_classifier.joblib"
    joblib.dump(artifact_package, model_save_path)
    print(f"\n[SUCCESS] Exported production model package to: {model_save_path}")


if __name__ == "__main__":
    run_pipeline()
