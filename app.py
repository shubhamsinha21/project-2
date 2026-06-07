import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model Files
# -----------------------------
try:
    model = joblib.load("heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("AI-powered clinical decision support system (educational use only)")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

    Gender = st.selectbox("Gender", ["male", "female"])

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["typical", "atypical", "non-anginal", "asymptomatic"]
    )

    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)

    cholesterol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fasting_bp = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "left ventricular hypertrophy"]
    )

    max_heart_rate = st.slider("Max Heart Rate", 60, 220, 150)

    exercise_angina = st.selectbox("Exercise Induced Angina", [0, 1])

    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

    st_slope = st.selectbox(
        "ST Slope",
        ["upsloping", "flat", "downsloping"]
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Risk", use_container_width=True):

    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "MaxHR": max_heart_rate,
        "Oldpeak": oldpeak,
        "FastingBS": fasting_bp,
        "Gender_" + Gender: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + str(exercise_angina): 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([input_data])

    # align columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]

    low_risk_prob = probability[0] * 100
    high_risk_prob = probability[1] * 100

    st.divider()

    # -----------------------------
    # Result Section
    # -----------------------------
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.metric("Risk Score", f"{high_risk_prob:.2f}%")
        st.progress(int(high_risk_prob))

    else:
        st.success("✅ Low Risk of Heart Disease")
        st.metric("Safety Score", f"{low_risk_prob:.2f}%")
        st.progress(int(low_risk_prob))

    st.divider()

    # -----------------------------
    # FEATURE IMPORTANCE SECTION (NEW)
    # -----------------------------
    st.subheader("📌 What influenced this prediction?")

    feature_names = columns

    # extract importance
    if hasattr(model, "coef_"):
        importance = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_names))

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df["AbsImportance"] = importance_df["Importance"].abs()
    importance_df = importance_df.sort_values("AbsImportance", ascending=False).head(10)

    # Plot
    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"][::-1], importance_df["AbsImportance"][::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Influencing Features")

    st.pyplot(fig)

    st.markdown("### Top contributing factors")
    st.dataframe(importance_df[["Feature", "Importance"]])

    st.divider()

    # -----------------------------
    # Probability Summary
    # -----------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Low Risk Probability", f"{low_risk_prob:.2f}%")

    with c2:
        st.metric("High Risk Probability", f"{high_risk_prob:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Educational project only. Not a medical diagnosis tool."
)