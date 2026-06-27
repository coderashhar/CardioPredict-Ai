import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load Model Artifacts ─────────────────────────────────────────────────────
model = joblib.load("Logistic_Regression_Heart_Model.pkl")
scaler = joblib.load("scaler.pkl")
expected_cols = joblib.load("columns.pkl")

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #1b2838 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        position: relative;
    }

    .heart-icon {
        font-size: 4rem;
        display: inline-block;
        animation: heartbeat 1.4s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(255, 75, 75, 0.6));
    }

    @keyframes heartbeat {
        0%   { transform: scale(1); }
        14%  { transform: scale(1.15); }
        28%  { transform: scale(1); }
        42%  { transform: scale(1.15); }
        70%  { transform: scale(1); }
        100% { transform: scale(1); }
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff4b4b, #ff8a80, #ff4b4b);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 4s ease infinite;
        margin: 0.5rem 0 0.3rem 0;
        letter-spacing: -1px;
    }

    @keyframes gradient-shift {
        0%   { background-position: 0% center; }
        50%  { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #8899aa;
        font-weight: 400;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    .hero-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #ff4b4b, transparent);
        margin: 1rem auto;
        border-radius: 2px;
    }

    /* ── Glassmorphism Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.6rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .glass-card:hover {
        border-color: rgba(255, 75, 75, 0.2);
        box-shadow: 0 8px 32px rgba(255, 75, 75, 0.08);
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .card-icon {
        font-size: 1.4rem;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: rgba(255, 75, 75, 0.1);
    }

    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    /* ── Streamlit Widget Overrides ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #ff4b4b, #ff8a80) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #ff4b4b !important;
        border: 3px solid #fff !important;
        box-shadow: 0 0 12px rgba(255, 75, 75, 0.5) !important;
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }

    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }

    /* ── Predict Button ── */
    .stButton > button {
        width: 100%;
        padding: 0.9rem 2rem;
        font-size: 1.15rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #fff;
        background: linear-gradient(135deg, #ff4b4b 0%, #c0392b 100%);
        border: none;
        border-radius: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(255, 75, 75, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(255, 75, 75, 0.5);
        background: linear-gradient(135deg, #ff6b6b 0%, #e74c3c 100%);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Result Cards ── */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeSlideUp 0.6s ease-out;
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-danger {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.08));
        border: 1px solid rgba(231, 76, 60, 0.3);
    }

    .result-safe {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.08));
        border: 1px solid rgba(46, 204, 113, 0.3);
    }

    .result-icon {
        font-size: 3.5rem;
        margin-bottom: 0.6rem;
    }

    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .result-danger .result-title { color: #e74c3c; }
    .result-safe .result-title   { color: #2ecc71; }

    .result-text {
        font-size: 1rem;
        color: #aab;
        line-height: 1.7;
        max-width: 500px;
        margin: 0 auto;
    }

    /* ── Confidence Bar ── */
    .confidence-container {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .confidence-label {
        font-size: 0.8rem;
        color: #889;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .confidence-bar-bg {
        width: 100%;
        height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        overflow: hidden;
    }
    .confidence-bar-fill-danger {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #e74c3c, #ff6b6b);
        animation: fillBar 1s ease-out;
    }
    .confidence-bar-fill-safe {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #2ecc71, #58d68d);
        animation: fillBar 1s ease-out;
    }
    @keyframes fillBar {
        from { width: 0%; }
    }
    .confidence-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .confidence-value.danger { color: #e74c3c; }
    .confidence-value.safe   { color: #2ecc71; }

    /* ── Info Badges ── */
    .info-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .info-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.04);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.82rem;
        color: #8899aa;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Section Titles ── */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #667;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 2rem 0 0.8rem 0;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.06);
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #556;
        font-size: 0.78rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 3rem;
    }

    /* ── Hide default label for styled sections ── */
    .block-container {
        padding-top: 1rem !important;
    }

    /* ── Pulse ring around heart ── */
    .pulse-ring {
        display: inline-block;
        position: relative;
    }
    .pulse-ring::before,
    .pulse-ring::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 70px;
        height: 70px;
        border-radius: 50%;
        border: 2px solid rgba(255, 75, 75, 0.3);
        transform: translate(-50%, -50%);
        animation: pulse-ring-anim 2s ease-out infinite;
    }
    .pulse-ring::after {
        animation-delay: 0.6s;
    }
    @keyframes pulse-ring-anim {
        0%   { width: 70px; height: 70px; opacity: 1; }
        100% { width: 140px; height: 140px; opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)


# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="pulse-ring">
        <div class="heart-icon">❤️</div>
    </div>
    <div class="hero-title">CardioPredict AI</div>
    <div class="hero-subtitle">
        AI-powered cardiac risk assessment using machine learning
    </div>
    <div class="hero-divider"></div>
    <div class="info-row">
        <div class="info-badge">🧠 Logistic Regression Model</div>
        <div class="info-badge">📊 918 Patient Dataset</div>
        <div class="info-badge">⚡ Instant Prediction</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Input Form ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-title">
    <span>Patient Information</span>
    <div class="section-line"></div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">👤</div>
            <div class="card-title">Demographics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    age = st.slider("🎂 Age", 18, 100, 40)
    sex = st.selectbox("⚧ Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")
    chest_pain = st.selectbox(
        "💢 Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"],
        format_func=lambda x: x.title(),
    )
    fasting_bs = st.selectbox("🩸 Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">🫀</div>
            <div class="card-title">Cardiac Metrics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    resting_bp = st.number_input("🩺 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("🧪 Cholesterol (mg/dl)", 100, 600, 200)
    max_hr = st.slider("💓 Max Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)

with col3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-header">
            <div class="card-icon">📋</div>
            <div class="card-title">ECG & Exercise</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    resting_ecg = st.selectbox(
        "📈 Resting ECG",
        ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"],
        format_func=lambda x: x.title(),
    )
    exercise_angina = st.selectbox("🏃 Exercise Induced Angina", ["No", "Yes"])
    st_slope = st.selectbox(
        "📊 ST Slope",
        ["upsloping", "flat", "downsloping"],
        format_func=lambda x: x.title(),
    )


# ─── Predict Button ──────────────────────────────────────────────────────────
st.markdown("")  # spacer
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict_clicked = st.button("❤️  Analyze Heart Health  ❤️", use_container_width=True)


# ─── Prediction Logic & Results ───────────────────────────────────────────────
if predict_clicked:
    # Build input dataframe
    raw_input = {
        "RestingBP": resting_bp,
        "Age": age,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "Cholesterol": cholesterol,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ST_Slope_" + st_slope: 1,
        "ExerciseAngina_" + exercise_angina: 1,
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = probabilities[prediction] * 100

    st.markdown("""
    <div class="section-title">
        <span>Analysis Results</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    if prediction == 1:
        risk_pct = probabilities[1] * 100
        st.markdown(f"""
        <div class="result-card result-danger">
            <div class="result-icon">⚠️</div>
            <div class="result-title">Heart Disease Risk Detected</div>
            <div class="result-text">
                Based on the health parameters you provided, our model indicates a 
                <strong>potential risk</strong> of heart disease. We strongly recommend 
                consulting a healthcare professional for a comprehensive evaluation.
            </div>
            <div class="confidence-container">
                <div class="confidence-label">Model Confidence</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill-danger" style="width: {confidence:.1f}%;"></div>
                </div>
                <div class="confidence-value danger">{confidence:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <div class="result-icon">✅</div>
            <div class="result-title">Low Risk — Heart Looks Healthy!</div>
            <div class="result-text">
                Great news! Based on the health parameters you provided, our model 
                suggests a <strong>low risk</strong> of heart disease. Keep maintaining 
                a healthy lifestyle with regular exercise and a balanced diet.
            </div>
            <div class="confidence-container">
                <div class="confidence-label">Model Confidence</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill-safe" style="width: {confidence:.1f}%;"></div>
                </div>
                <div class="confidence-value safe">{confidence:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    ⚕️ <strong>Disclaimer:</strong> This tool is for educational purposes only and is 
    not a substitute for professional medical advice, diagnosis, or treatment.<br>
    Built with ❤️ using Streamlit & Scikit-Learn
</div>
""", unsafe_allow_html=True)
