
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
    section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #374151;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/*header {visibility: hidden;}*/

/* Main Title */
.main-title {
    text-align: center;
    color: white;
    font-size: 48px;
    font-weight: 700;
    margin-bottom: 10px;
}
/* Subtitle */
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 20px;
    margin-bottom: 25px;
}
/* KPI Cards */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.10);
}
/* Result Cards */
.approved {
    background: rgba(16,185,129,0.20);
    border: 1px solid rgba(16,185,129,0.40);
    padding: 25px;
    border-radius: 15px;
    color: #10b981;
    font-size: 28px;
    text-align: center;
    font-weight: bold;
}

.rejected {
    background: rgba(239,68,68,0.20);
    border: 1px solid rgba(239,68,68,0.40);
    padding: 25px;
    border-radius: 15px;
    color: #ef4444;
    font-size: 28px;
    text-align: center;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "Model" / "loan_approval_xgboost.pkl")
scaler = joblib.load(BASE_DIR / "Model" / "scaler.pkl")

# SIDEBAR
with st.sidebar:
    st.title("🏦 Project Dashboard")

    st.markdown("### Developer")
    st.write("Anvesha Srivastava")

    st.markdown("### Model")
    st.write("XGBoost Classifier")

    st.metric("Accuracy", "98.36%")
    st.metric("ROC-AUC", "99.86%")

    st.success("🟢 Production Ready")


st.markdown("""
<div class="main-title">
🏦 Loan Approval Prediction System
</div>

<div class="subtitle">
AI-Powered Banking Loan Eligibility Checker
</div>
""", unsafe_allow_html=True)

st.info(
    "💡 Enter applicant details and let the AI model evaluate loan eligibility instantly."
)


# KPI CARDS

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h2>98.36%</h2>
        <p>Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h2>XGBoost</h2>
        <p>Best Model</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h2>99.86%</h2>
        <p>ROC-AUC</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(
    ["📋 Prediction", "📊 Analytics", "ℹ️ About Project"]
)

# TAB 1 - PREDICTION

with tab1:
    st.subheader("Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=0
        )
        education = st.selectbox(
            "Education",
            ["Not Graduate", "Graduate"]
        )
        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )
        income_annum = st.number_input(
            "Annual Income (₹)",
            min_value=0
        )
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0
        )
        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1
        )
    with col2:
        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=700
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value (₹)",
            min_value=0
        )
        commercial_assets_value = st.number_input(
            "Commercial Assets Value (₹)",
            min_value=0
        )
        luxury_assets_value = st.number_input(
            "Luxury Assets Value (₹)",
            min_value=0
        )
        bank_asset_value = st.number_input(
            "Bank Asset Value (₹)",
            min_value=0
        )
    if st.button("🔍 Predict Loan Status"):

        education_encoded = 1 if education == "Graduate" else 0
        self_employed_encoded = 1 if self_employed == "Yes" else 0

        total_assets = (
            residential_assets_value
            + commercial_assets_value
            + luxury_assets_value
            + bank_asset_value
        )
        st.subheader("📄 Applicant Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(f"**Annual Income:** ₹{income_annum:,}")
            st.write(f"**Loan Amount:** ₹{loan_amount:,}")

        with summary_col2:
            st.write(f"**CIBIL Score:** {cibil_score}")
            st.write(f"**Total Assets:** ₹{total_assets:,}")

        sample = pd.DataFrame({

            "no_of_dependents":[no_of_dependents],
            "education":[education_encoded],
            "self_employed":[self_employed_encoded],
            "income_annum":[income_annum],
            "loan_amount":[loan_amount],
            "loan_term":[loan_term],
            "cibil_score":[cibil_score],
            "residential_assets_value":[residential_assets_value],
            "commercial_assets_value":[commercial_assets_value],
            "luxury_assets_value":[luxury_assets_value],
            "bank_asset_value":[bank_asset_value],
            "total_assets":[total_assets]

        })

        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled)[0][1]

        st.subheader("📈 Approval Probability")

        st.progress(int(probability * 100))

        st.write(
            f"### {probability*100:.2f}% Probability of Approval"
        )

        if prediction == 0:

            st.markdown("""
            <div class="approved">
            ✅ LOAN APPROVED
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

        else:

            st.markdown("""
            <div class="rejected">
            ❌ LOAN REJECTED
            </div>
            """, unsafe_allow_html=True)
        

# TAB 2 - ANALYTICS

with tab2:

    st.subheader("Feature Importance Analysis")

    features = [
        "CIBIL Score",
        "Loan Term",
        "Loan Amount",
        "Income",
        "Bank Assets",
        "Luxury Assets",
        "Residential Assets",
        "Self Employed",
        "Dependents",
        "Commercial Assets",
        "Total Assets",
        "Education"
    ]

    importance = [
        67.99,
        19.81,
        2.58,
        2.27,
        1.22,
        1.09,
        1.07,
        1.04,
        0.95,
        0.86,
        0.58,
        0.51
    ]

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(features, importance)
    ax.set_title("Feature Importance using XGBoost")
    ax.set_xlabel("Importance (%)")

    st.pyplot(fig)

# TAB 3 - ABOUT PROJECT

with tab3:

    st.subheader("Project Overview")

    st.write("""
    This Loan Approval Prediction System uses Machine Learning
    to determine whether a loan application should be approved
    or rejected based on applicant financial information.

    Features Used:
    - Income
    - Loan Amount
    - Loan Term
    - CIBIL Score
    - Asset Information
    - Employment Status
    - Dependents

    Models Trained:
    - Logistic Regression
    - Random Forest
    - XGBoost

    Best Model:
    XGBoost Classifier
    """)

    st.success(
        "Final Model Selected: XGBoost (Accuracy: 98.36%)"
    )