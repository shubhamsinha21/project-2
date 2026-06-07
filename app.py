import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from io import BytesIO
from reportlab.pdfgen import canvas

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Risk AI Dashboard",
    page_icon="🫀",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# HEADER (HOSPITAL STYLE)
# -----------------------------
st.markdown("""
<div style="background-color:#0f172a;padding:20px;border-radius:12px;text-align:center">
    <h1 style="color:white;">🫀 Heart Risk AI Dashboard </h1>
    <p style="color:#cbd5e1;">AI-powered heart disease risk analysis dashboard</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# INPUT SECTION
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Patient Age", 18, 100, 45)
    gender = st.selectbox("Gender", ["male", "female"])
    chest_pain = st.selectbox("Chest Pain Type",
        ["typical", "atypical", "non-anginal", "asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol Level", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    resting_ecg = st.selectbox("Resting ECG",
        ["Normal", "ST", "left ventricular hypertrophy"])
    max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope",
        ["upsloping", "flat", "downsloping"])

st.divider()

# -----------------------------
# PDF FUNCTION
# -----------------------------
def generate_pdf(data_dict):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)

    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, 800, "Heart Disease Risk Report")

    p.setFont("Helvetica", 11)
    y = 760

    for k, v in data_dict.items():
        p.drawString(50, y, f"{k}: {v}")
        y -= 20

    p.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Analyze Risk", use_container_width=True):

    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "FastingBS": fasting_bs,
        "Gender_" + gender: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + str(exercise_angina): 1,
        "ST_Slope_" + st_slope: 1
    }

    df = pd.DataFrame([input_data])

    for c in columns:
        if c not in df.columns:
            df[c] = 0

    df = df[columns]

    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    risk = prob[1] * 100

    # -----------------------------
    # RISK CLASSIFICATION
    # -----------------------------
    if risk < 30:
        status = "LOW RISK"
        color = "🟢"
        box_color = "#16a34a"
    elif risk < 70:
        status = "MODERATE RISK"
        color = "🟡"
        box_color = "#facc15"
    else:
        status = "HIGH RISK"
        color = "🔴"
        box_color = "#ef4444"

    st.divider()

    # -----------------------------
    # PATIENT DASHBOARD (HOSPITAL CARD STYLE)
    # -----------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style="padding:20px;border-radius:15px;background-color:#111827;color:white">
            <h3>🧑 Patient Card</h3>
            <p><b>Age:</b> {age}</p>
            <p><b>Gender:</b> {gender}</p>
            <p><b>Chest Pain:</b> {chest_pain}</p>
            <p><b>BP:</b> {resting_bp}</p>
            <p><b>Cholesterol:</b> {cholesterol}</p>
            <p><b>Max HR:</b> {max_hr}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="padding:25px;border-radius:15px;background-color:{box_color};color:white;text-align:center">
            <h2>🫀 {status}</h2>
            <h1 style="font-size:50px">{risk:.2f}%</h1>
            <p>Heart Disease Probability</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(risk))
        st.caption("Risk Meter (0 → Safe, 100 → Critical)")

    st.divider()

    # -----------------------------
    # RISK CHART
    # -----------------------------
    st.subheader("📊 Risk Visualization")

    st.bar_chart(pd.DataFrame({
        "Risk": [risk]
    }))

    st.divider()

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("🧠 Clinical Indicators Impact")

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.zeros(len(columns))

    feat_df = pd.DataFrame({
        "Feature": columns,
        "Impact": importance
    }).sort_values("Impact", ascending=False).head(8)

    st.bar_chart(feat_df.set_index("Feature"))

    st.divider()

    # -----------------------------
    # PDF REPORT
    # -----------------------------
    st.subheader("📄 Medical Report")

    report_data = {
        "Age": age,
        "Gender": gender,
        "Chest Pain": chest_pain,
        "BP": resting_bp,
        "Cholesterol": cholesterol,
        "Max HR": max_hr,
        "Oldpeak": oldpeak,
        "Risk Status": status,
        "Risk Score": f"{risk:.2f}%",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    pdf = generate_pdf(report_data)

    st.download_button(
        "📥 Download Report",
        pdf,
        file_name="heart_risk_report.pdf",
        mime="application/pdf"
    )

# -----------------------------
# FOOTER
# -----------------------------
st.caption("🫀 Heart Risk AI v2 | Made with ❤️ by Shubham Sinha")