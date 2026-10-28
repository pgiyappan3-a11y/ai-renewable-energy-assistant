from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
DATA_PATH = ROOT / "data" / "simulated_renewable_energy.csv"

st.set_page_config(page_title="Renewable Energy Assistant", page_icon="☀️", layout="wide")
st.title("☀️ AI-Based Renewable Energy Source Selection Assistant")
st.write("Educational prototype that compares solar and wind suitability from site and weather inputs.")
st.warning("Important: This demo uses a simulated educational dataset. Results are indicative only and are not a substitute for a professional feasibility study.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

with st.sidebar:
    st.header("Enter site conditions")
    irradiance = st.slider("Solar irradiance (W/m²)", 0, 1100, 600, 10)
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 25.0, 6.0, 0.5)
    temperature = st.slider("Temperature (°C)", 5.0, 48.0, 28.0, 0.5)
    humidity = st.slider("Humidity (%)", 10, 100, 60)
    cloud = st.slider("Cloud cover (%)", 0, 100, 30)
    hour = st.slider("Hour of day", 0, 23, 12)
    month = st.selectbox("Month", list(range(1,13)), index=5)

input_df = pd.DataFrame([{
    "solar_irradiance_wm2": irradiance,
    "wind_speed_ms": wind_speed,
    "temperature_c": temperature,
    "humidity_pct": humidity,
    "cloud_cover_pct": cloud,
    "hour": hour,
    "month": month
}])
prediction = model.predict(input_df)[0]
st.subheader("AI recommendation")
if prediction == "Solar":
    st.success("Recommended category: SOLAR")
elif prediction == "Wind":
    st.success("Recommended category: WIND")
elif prediction == "Both":
    st.info("Recommended category: BOTH — compare local costs and constraints")
else:
    st.warning("Recommended category: LOW POTENTIAL — collect more site data")

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(input_df)[0]
    classes = model.classes_
    prob_df = pd.DataFrame({"Category": classes, "Model score (%)": (probs*100).round(1)})
    st.write("Model class scores (not guaranteed real-world probabilities)")
    st.bar_chart(prob_df.set_index("Category"))

c1,c2 = st.columns(2)
with c1:
    st.metric("Solar irradiance", f"{irradiance} W/m²")
    st.metric("Wind speed", f"{wind_speed:.1f} m/s")
with c2:
    st.metric("Temperature", f"{temperature:.1f} °C")
    st.metric("Cloud cover", f"{cloud}%")

st.subheader("Dataset exploration")
left,right = st.columns(2)
with left:
    st.write("Recommended source counts (simulated data)")
    st.bar_chart(data["recommended_source"].value_counts())
with right:
    st.write("Solar vs wind potential indices")
    st.scatter_chart(data, x="solar_potential_index", y="wind_potential_index", color="recommended_source")

st.caption("Use only for learning and demonstration. Actual project decisions require long-term measured weather data, terrain, grid access, land, costs, and professional engineering analysis.")
