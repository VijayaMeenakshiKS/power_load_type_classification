"""
Streamlit Web Dashboard: Power System Load Type Predictor
==========================================================
Interactive interface for real-time load classification,
physics-based power factor monitoring, and industrial tariff risk advisory.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

st.set_page_config(
    page_title="Power Load Classification & Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #2b5876;
    }
    .badge-light {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .badge-maximum {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    model_path = Path("models/best_load_classifier.joblib")
    if model_path.exists():
        try:
            package = joblib.load(model_path)
            return package
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None


package = load_trained_model()

# Header
st.title("⚡ Power System Load Type Classification & Energy Intelligence")
st.markdown("""
Predicting factory **Load Type** (*Light Load*, *Medium Load*, *Maximum Load*) using 
**telemetry sensors**, **time dynamics**, and **AC Power Triangle physics**.
""")

# Sidebar inputs
st.sidebar.header("🛠️ Factory Sensor Telemetry")

hour = st.sidebar.slider("Hour of Day (24-hr format)", 0, 23, 14, help="Shift time")
minute = st.sidebar.slider("Minute of Hour", 0, 59, 30, step=15)
is_weekend = st.sidebar.radio("Day Type", options=["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"]) == "Weekend (Sat-Sun)"

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Electrical Power Readings")
usage_kwh = st.sidebar.number_input("Active Energy Consumption (Usage_kWh)", min_value=0.0, max_value=250.0, value=65.4, step=0.5)
lagging_q = st.sidebar.number_input("Lagging Reactive Power (kVarh)", min_value=0.0, max_value=120.0, value=28.5, step=0.5, help="Inductive load (motors/compressors)")
leading_q = st.sidebar.number_input("Leading Reactive Power (kVarh)", min_value=0.0, max_value=80.0, value=0.0, step=0.1, help="Capacitive load (power factor correction banks)")
co2_ppm = st.sidebar.number_input("CO2 Concentration (ppm or tCO2)", min_value=0.0, max_value=0.5, value=0.03, step=0.005)

# Derived Feature Computation
nsm = hour * 3600 + minute * 60
sin_nsm = np.sin(2 * np.pi * nsm / 86400.0)
cos_nsm = np.cos(2 * np.pi * nsm / 86400.0)
sin_hour = np.sin(2 * np.pi * hour / 24.0)
cos_hour = np.cos(2 * np.pi * hour / 24.0)

net_reactive = lagging_q - leading_q
total_reactive = lagging_q + leading_q
apparent_power = np.sqrt(usage_kwh**2 + net_reactive**2)
power_factor = usage_kwh / (apparent_power + 1e-6)
power_factor = float(np.clip(power_factor, 0.0, 1.0))
reactive_active_ratio = net_reactive / (usage_kwh + 1e-6)
carbon_intensity = co2_ppm / (usage_kwh + 1e-6)

# Main Dashboard Layout
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("⏰ Seconds from Midnight (NSM)", f"{nsm:,} s", delta=f"{hour:02d}:{minute:02d}")
with col2:
    st.metric("⚡ Apparent Power (S)", f"{apparent_power:.2f} kVAh")
with col3:
    st.metric("📊 Operating Power Factor", f"{power_factor:.3f}", delta="Optimal >= 0.90" if power_factor >= 0.9 else "Low PF Penalty Risk", delta_color="normal" if power_factor >= 0.9 else "inverse")
with col4:
    st.metric("🌱 CO2 / Energy Ratio", f"{carbon_intensity:.4f}")

st.markdown("---")

# Prediction Column & Physics Column
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("🎯 Machine Learning Load Type Prediction")
    
    if package is None:
        st.warning("""
        ⚠️ **Trained model artifact not found!**
        
        To generate and save the production model:
        1. Run the Jupyter Notebook cells in `notebooks/power_load_classification.ipynb`, or
        2. Execute `python train_model.py` in your terminal.
        
        *(Falling back to heuristic power rule for demo visualization)*
        """)
        # Fallback heuristic
        if usage_kwh < 20 or is_weekend:
            pred_class = "Light_Load"
            probs = [0.92, 0.06, 0.02]
        elif usage_kwh < 60:
            pred_class = "Medium_Load"
            probs = [0.08, 0.81, 0.11]
        else:
            pred_class = "Maximum_Load"
            probs = [0.02, 0.13, 0.85]
        classes = ["Light_Load", "Medium_Load", "Maximum_Load"]
        best_name = "Rule-based Electrical Heuristic"
    else:
        best_model = package["model"]
        best_name = package.get("model_name", "Classifier")
        feature_cols = package["feature_cols"]
        classes = package["classes"]
        scaler = package["scaler"]

        # Construct input dictionary matching trained features
        input_data = {
            'Usage_kWh': usage_kwh,
            'Lagging_Current_Reactive_Power_kVarh': lagging_q,
            'Leading_Current_Reactive_Power_kVarh': leading_q,
            'CO2': co2_ppm,
            'NSM': nsm,
            'Hour': hour,
            'DayOfWeek': 5 if is_weekend else 2,
            'Is_Weekend': int(is_weekend),
            'sin_NSM': sin_nsm,
            'cos_NSM': cos_nsm,
            'sin_Hour': sin_hour,
            'cos_Hour': cos_hour,
            'Net_Reactive_Power_kVarh': net_reactive,
            'Total_Reactive_Power_kVarh': total_reactive,
            'Apparent_Power_kVAh': apparent_power,
            'Operating_Power_Factor': power_factor,
            'Reactive_Active_Ratio': reactive_active_ratio,
            'Carbon_Intensity': carbon_intensity,
            'Usage_Lag_1': usage_kwh,
            'Usage_Delta_1': 0.0,
            'Usage_Rolling_Mean_4': usage_kwh
        }

        # Build feature vector matching training schema
        feature_vec = pd.DataFrame([{col: input_data.get(col, 0.0) for col in feature_cols}])
        
        if scaler is not None:
            feature_vec_proc = scaler.transform(feature_vec)
        else:
            feature_vec_proc = feature_vec

        pred_idx = best_model.predict(feature_vec_proc)[0]
        pred_class = classes[pred_idx]
        
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(feature_vec_proc)[0]
        else:
            probs = [1.0 if i == pred_idx else 0.0 for i in range(len(classes))]

    # Render badge
    st.write(f"**Serving Model:** `{best_name}`")
    if pred_class == "Light_Load":
        st.markdown('<div class="badge-light">🟢 Predicted: LIGHT LOAD</div>', unsafe_allow_html=True)
        st.success("Off-peak operational phase. Low electricity tariff rate applies.")
    elif pred_class == "Medium_Load":
        st.markdown('<div class="badge-medium">🟡 Predicted: MEDIUM LOAD</div>', unsafe_allow_html=True)
        st.warning("Normal active shift. Standard industrial tariff rate applies.")
    else:
        st.markdown('<div class="badge-maximum">🔴 Predicted: MAXIMUM LOAD</div>', unsafe_allow_html=True)
        st.error("🚨 HIGH PEAK DEMAND TIER! Utility peak demand penalty applies. Consider dispatching backup or load shedding.")

    # Probability Bars
    st.write("#### Class Probabilities:")
    prob_df = pd.DataFrame({
        "Load Category": classes,
        "Probability": probs
    })
    st.bar_chart(prob_df.set_index("Load Category"))

with right_col:
    st.subheader("📐 AC Power Triangle Breakdown")
    st.info(f"""
    **Active Real Power ($P$):** `{usage_kwh:.2f} kWh`  
    **Net Reactive Power ($Q$):** `{net_reactive:.2f} kVarh`  
    **Apparent Power ($S = \\sqrt{{P^2 + Q^2}}$):** `{apparent_power:.2f} kVAh`  
    **Power Factor ($\\cos\\phi$):** `{power_factor:.3f}`  
    """)

    st.subheader("💡 Industrial Decision Support & Tariff Risk")
    if pred_class == "Maximum_Load":
        st.markdown("""
        > **Recommended Action:**  
        > * Stagger non-critical motor drives and compressors.  
        > * Check capacitor banks to keep Power Factor above 0.95.  
        > * Defer high-temperature smelting/furnace cycles by 45 minutes to drop into Medium tier.
        """)
    elif power_factor < 0.90:
        st.markdown("""
        > **Power Factor Alert:**  
        > Current Power Factor is below 0.90. The utility company will levy low-power-factor surcharges. Switch in local capacitor banks.
        """)
    else:
        st.markdown("""
        > **Optimal Operational State:**  
        > Load parameters are well within balanced grid margins.
        """)

st.markdown("---")
st.caption("Power System Load Type Classification Project | Antigravity AI Engine")
