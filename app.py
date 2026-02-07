import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import os

os.environ["SHAP_HIDE_WARNINGS"] = "1"

# ---------- Page setup ----------
st.set_page_config(
    page_title="Diabetes Risk Awareness Tool",
    layout="centered"
)

st.title("🩺 Diabetes Risk Awareness Tool")

st.write(
    "This tool helps you understand your diabetes risk level "
    "using general health information."
)

st.info(
    "ℹ️ This is an awareness tool, not a medical diagnosis. "
    "It combines machine learning with medical safety checks "
    "to avoid misleading results."
)

# ---------- Load trained model ----------
model = joblib.load("diabetes_model.pkl")

# ---------- Load data for explanation ----------
df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)

explainer = shap.Explainer(
    model.named_steps["model"],
    X
)

# ---------- User Inputs ----------
st.subheader("🧍 Personal Information")

age = st.number_input("Age (years)", 1, 120, 30)
pregnancies = st.number_input(
    "Number of pregnancies (0 if not applicable)",
    0, 20, 0
)

st.subheader("📏 Body Measurements")

height_cm = st.number_input("Height (cm)", 100, 220, 170)
weight_kg = st.number_input("Weight (kg)", 30, 200, 70)

# ---------- BMI Calculation ----------
height_m = height_cm / 100
bmi = round(weight_kg / (height_m ** 2), 2)
st.caption(f"📊 Calculated BMI: **{bmi}**")

st.subheader("🩸 Health Measurements")

glucose = st.number_input("Blood sugar level (glucose)", 50, 300, 120)
st.caption("💡 Normal fasting glucose is usually between 70–100 mg/dL")

blood_pressure = st.number_input(
    "Blood pressure (lower number)",
    40, 200, 80
)

skin_thickness = st.number_input(
    "Skin thickness (leave default if unknown)",
    0, 100, 20
)

insulin = st.number_input(
    "Insulin level (leave default if unknown)",
    0, 900, 80
)

st.subheader("🧬 Background Information")

dpf = st.number_input(
    "Family history of diabetes",
    0.0, 3.0, 0.0
)

# ---------- Prediction ----------
if st.button("Check Diabetes Risk"):
    user_data = np.array([[ 
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        dpf,
        bmi,
        age
    ]])

    # ---------- SAFETY RULE 1: Medically Low Risk ----------
    medically_low_risk = (
        glucose < 100 and
        dpf == 0.0 and
        bmi < 32
    )

    probability = model.predict_proba(user_data)[0][1] * 100

    # ---------- SAFETY RULE 2: Age Bias Guard ----------
    if probability > 70 and glucose < 110 and dpf == 0.0:
        probability = 55  # force MODERATE instead of HIGH

    # ---------- Risk Band Logic ----------
    if medically_low_risk:
        risk_level = "LOW"
    elif probability < 35:
        risk_level = "LOW"
    elif probability < 70:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    # ---------- Result ----------
    st.subheader("📊 Risk Assessment")

    st.markdown("### 📈 Risk Meter")

    if risk_level == "LOW":
        st.progress(0.25)
        st.success("🟢 Low diabetes risk")
        st.caption("Low risk zone")

    elif risk_level == "MODERATE":
        st.progress(0.60)
        st.warning("🟡 Moderate diabetes risk")
        st.caption("Moderate risk zone")

    else:
        st.progress(0.90)
        st.error("🔴 High diabetes risk")
        st.caption("High risk zone")

    # ---------- Text Explanation ----------
    if medically_low_risk:
        st.write(
            "Your blood sugar level, body weight range, and lack of family history "
            "suggest a low diabetes risk based on common medical guidelines."
        )
    else:
        st.write(
            "This result is based on patterns observed in historical health data. "
            "It does not confirm or rule out diabetes."
        )

        # ---------- Explainable AI ----------
        st.subheader("🔍 What influenced this result?")

        shap_values = explainer(user_data)
        impacts = shap_values.values[0]

        explanation = sorted(
            zip(X.columns, impacts),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, value in explanation[:3]:
            if value > 0:
                st.info(f"• {feature} had a stronger influence")
            else:
                st.info(f"• {feature} had less influence")

    # ---------- Guidance ----------
    st.subheader("📌 What can you do next?")

    st.write(
        "- Maintain a balanced diet and regular physical activity\n"
        "- Monitor blood sugar levels if advised\n"
        "- Consult a healthcare professional for clarity"
    )

    st.caption(
        "⚠️ This tool is for awareness and learning purposes only."
    )
