import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CreditWise Loan Approval",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.title {font-size:40px;font-weight:700;color:#1f4eff}
.subtitle {font-size:18px;color:#555}
.section {background:#f8f9ff;padding:18px;border-radius:14px;margin-bottom:20px}
.approved {background:#e7f7ef;padding:20px;border-radius:14px}
.rejected {background:#fdeaea;padding:20px;border-radius:14px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💳 CreditWise Loan Approval System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Loan eligibility prediction with explainable ML</div>', unsafe_allow_html=True)
st.divider()

model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("💰 Financial Details")
    Applicant_Income = st.number_input("Applicant Income (₹)", 0, value=5000, step=1000)
    Coapplicant_Income = st.number_input("Co-applicant Income (₹)", 0, value=0, step=1000)
    Loan_Amount = st.number_input("Loan Amount (₹)", 0, value=100000, step=10000)
    Loan_Term = st.number_input("Loan Term (months)", 6, 360, value=120)
    Savings = st.number_input("Savings (₹)", 0, value=20000, step=5000)
    Collateral_Value = st.number_input("Collateral Value (₹)", 0, value=50000, step=5000)
    Existing_Loans = st.number_input("Existing Loans", 0, 10, value=0)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("👤 Personal & Loan Info")
    Age = st.number_input("Age", 18, 70, value=30)
    Dependents = st.number_input("Dependents", 0, 10, value=0)
    Credit_Score = st.number_input("Credit Score", 300, 900, value=650)
    Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    Marital_Status = st.selectbox("Marital Status", ["Single", "Married"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    Employer_Category = st.selectbox("Employer Category", ["Government", "Private", "Self-employed", "Other"])
    Loan_Purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal"])
    Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

if st.button("🔍 Check Loan Eligibility", use_container_width=True, type="primary"):

    DTI_Ratio = Loan_Amount / max(Applicant_Income + Coapplicant_Income, 1)

    features = {col: 0 for col in feature_names}
    features.update({
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Age": Age,
        "Dependents": Dependents,
        "Credit_Score": Credit_Score,
        "Existing_Loans": Existing_Loans,
        "DTI_Ratio": DTI_Ratio,
        "Savings": Savings,
        "Collateral_Value": Collateral_Value,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term
    })

    for key in [
        f"Employment_Status_{Employment_Status}",
        f"Marital_Status_{Marital_Status}",
        f"Loan_Purpose_{Loan_Purpose}",
        f"Education_Level_{Education_Level}",
        f"Employer_Category_{Employer_Category}"
    ]:
        if key in feature_names:
            features[key] = 1

    if Property_Area != "Rural" and f"Property_Area_{Property_Area}" in feature_names:
        features[f"Property_Area_{Property_Area}"] = 1

    if Gender == "Male" and "Gender_Male" in feature_names:
        features["Gender_Male"] = 1

    input_df = pd.DataFrame([features])[feature_names]
    input_scaled = scaler.transform(input_df.values)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else 0.5

    st.divider()
    st.subheader("📊 Loan Decision")

    if prediction == 1:
        st.markdown('<div class="approved">', unsafe_allow_html=True)
        st.success(f"### ✅ Loan Approved\nConfidence: {probability*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rejected">', unsafe_allow_html=True)
        st.error(f"### ❌ Loan Rejected\nConfidence: {(1-probability)*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📈 Approval Confidence")
    confidence = probability if prediction == 1 else (1 - probability)
    st.progress(confidence)

    st.subheader("💰 Financial Snapshot")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Income", f"₹{Applicant_Income + Coapplicant_Income:,}")
    c2.metric("DTI Ratio", f"{DTI_Ratio:.2f}")
    c3.metric("Savings Ratio", f"{Savings / max(Loan_Amount,1):.2f}")

    st.subheader("📊 Risk vs Strength Analysis")

    risk = {
        "High DTI": DTI_Ratio > 0.4,
        "Low Credit": Credit_Score < 600,
        "Low Savings": Savings < 0.1 * Loan_Amount,
        "Many Loans": Existing_Loans > 2
    }

    strength = {
        "Good Credit": Credit_Score >= 700,
        "Low DTI": DTI_Ratio < 0.3,
        "Strong Savings": Savings >= 0.2 * Loan_Amount,
        "No Loans": Existing_Loans == 0
    }

    viz_df = pd.DataFrame({
        "Factor": list(risk.keys()) + list(strength.keys()),
        "Impact": [-1 if v else 0 for v in risk.values()] + [1 if v else 0 for v in strength.values()]
    }).set_index("Factor")

    st.bar_chart(viz_df)

    st.subheader("⚖️ Risk vs Strength Balance")

    pie_df = pd.DataFrame({
        "Category": ["Risk Factors", "Strength Factors"],
        "Count": [sum(risk.values()), sum(strength.values())]
    })

    fig, ax = plt.subplots()
    ax.pie(pie_df["Count"], labels=pie_df["Category"], autopct="%1.0f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
