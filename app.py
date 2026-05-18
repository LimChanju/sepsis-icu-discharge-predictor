import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="ICU Early Discharge Predictor",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("LightGBM_20260517_160400.pkl")

model = load_model()
THRESHOLD = 0.519

st.title("ICU Early Discharge Predictor")
st.markdown(
    "Predicts the probability of **ICU discharge within 72 hours** for sepsis patients, "
    "using a LightGBM model trained on MIMIC-IV and externally validated on eICU-CRD."
)
st.divider()

st.subheader("Patient Variables (First 24 Hours of ICU Admission)")

col1, col2 = st.columns(2)

with col1:
    apsiii = st.number_input(
        "APS III Score", min_value=0, max_value=200, value=40,
        help="Acute Physiology Score III"
    )
    invasive_mv = st.radio(
        "Invasive Mechanical Ventilation",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        horizontal=True
    )
    heart_rate_max = st.number_input("Heart Rate Max (bpm)", min_value=20, max_value=300, value=100)
    mbp_min = st.number_input(
        "MBP Min (mmHg)", min_value=0, max_value=200, value=65,
        help="Minimum Mean Blood Pressure"
    )
    gcs_score = st.number_input(
        "GCS Score", min_value=3, max_value=15, value=15,
        help="Glasgow Coma Scale"
    )
    spo2_min = st.number_input("SpO₂ Min (%)", min_value=50, max_value=100, value=95)
    platelet_min = st.number_input("Platelet Min (×10³/µL)", min_value=0, max_value=2000, value=200)

with col2:
    sbp_min = st.number_input(
        "SBP Min (mmHg)", min_value=0, max_value=250, value=90,
        help="Minimum Systolic Blood Pressure"
    )
    sbp_max = st.number_input(
        "SBP Max (mmHg)", min_value=0, max_value=300, value=140,
        help="Maximum Systolic Blood Pressure"
    )
    respiratory_rate_min = st.number_input(
        "Respiratory Rate Min (breaths/min)", min_value=0, max_value=60, value=12
    )
    glucose_min = st.number_input("Glucose Min (mg/dL)", min_value=0, max_value=1000, value=100)
    sodium_max = st.number_input("Sodium Max (mEq/L)", min_value=100, max_value=200, value=140)
    chloride_max = st.number_input("Chloride Max (mEq/L)", min_value=60, max_value=160, value=105)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "apsiii": apsiii,
        "invasive_mechanical_ventilation": invasive_mv,
        "chloride_max": chloride_max,
        "heart_rate_max": heart_rate_max,
        "mbp_min": mbp_min,
        "gcs_score": gcs_score,
        "spo2_min": spo2_min,
        "platelet_min": platelet_min,
        "sbp_min": sbp_min,
        "respiratory_rate_min": respiratory_rate_min,
        "glucose_min": glucose_min,
        "sodium_max": sodium_max,
        "sbp_max": sbp_max,
    }])

    prob = model.predict_proba(input_df)[0][1]
    discharge_likely = prob >= THRESHOLD
    bar_color = "#27ae60" if discharge_likely else "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Probability of ICU Discharge within 72h", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, THRESHOLD * 100], "color": "#fdecea"},
                {"range": [THRESHOLD * 100, 100], "color": "#e9f7ef"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": THRESHOLD * 100,
            },
        }
    ))
    fig.update_layout(height=360, margin=dict(t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    if discharge_likely:
        st.success(
            f"**Early discharge likely** — Predicted probability: {prob:.1%}\n\n"
            "This patient may be a candidate for ICU-to-ward transfer within 72 hours."
        )
    else:
        st.error(
            f"**Early discharge unlikely** — Predicted probability: {prob:.1%}\n\n"
            "This patient may require continued ICU-level care beyond 72 hours."
        )

    st.caption(
        f"Decision threshold: {THRESHOLD} (Youden's index, derived from internal validation on MIMIC-IV). "
        "This tool is intended for research and clinical decision support only — not a substitute for clinical judgement."
    )

st.divider()
st.caption(
    "LightGBM model trained on MIMIC-IV · Validated on eICU-CRD · "
    "For research use only — not a substitute for clinical judgement"
)
