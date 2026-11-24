from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "best_model_metrics.json"

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📉", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("📉 Customer Churn Analysis & Prediction")
st.caption("Telecommunications customer retention analytics using the IBM Telco sample dataset.")

if not DATA_PATH.exists():
    st.warning("Cleaned dataset not found. Run `python -m src.train` first.")
    st.stop()
if not MODEL_PATH.exists():
    st.warning("Trained model not found. Run `python -m src.train` first.")
    st.stop()

df = load_data()
model = load_model()
df["ChurnLabel"] = df["Churn"].map({0: "Retained", 1: "Churned"})
churn_rate = df["Churn"].mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{len(df):,}")
c2.metric("Churn rate", f"{churn_rate:.1f}%")
c3.metric("Churned customers", f"{int(df['Churn'].sum()):,}")
c4.metric("Average tenure", f"{df['tenure'].mean():.1f} months")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Customer insights", "Predict churn", "Model results"])

with tab1:
    left, right = st.columns(2)
    with left:
        fig = px.pie(df, names="ChurnLabel", hole=0.55, title="Customer retention vs churn")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_contract = df.groupby("Contract", as_index=False)["Churn"].mean()
        by_contract["Churn rate (%)"] = by_contract["Churn"] * 100
        fig = px.bar(by_contract, x="Contract", y="Churn rate (%)",
                     title="Churn rate by contract type", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)
    st.info("This dashboard describes the churn labels in the supplied sample. It does not establish that a particular business action causes churn to decrease.")

with tab2:
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="tenure", color="ChurnLabel", barmode="overlay",
                           nbins=24, title="Tenure distribution by churn status")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.box(df, x="ChurnLabel", y="MonthlyCharges", color="ChurnLabel",
                     title="Monthly charges by churn status")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Churn rate by payment method")
    pay = df.groupby("PaymentMethod", as_index=False)["Churn"].mean()
    pay["Churn rate (%)"] = pay["Churn"] * 100
    st.plotly_chart(px.bar(pay, x="PaymentMethod", y="Churn rate (%)",
                           title="Churn rate by payment method", text_auto=".1f"),
                    use_container_width=True)

with tab3:
    st.subheader("Predict churn for a customer")
    with st.form("prediction_form"):
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        with b:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        with c:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=200.0, value=70.0)
            total = st.number_input("Total charges", min_value=0.0, max_value=20000.0,
                                    value=float(monthly * tenure))
        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        record = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total
        }])
        probability = float(model.predict_proba(record)[0, 1])
        prediction = int(probability >= 0.5)
        st.metric("Estimated churn probability", f"{probability:.1%}")
        if prediction:
            st.error("Model prediction: Higher churn risk (positive class).")
        else:
            st.success("Model prediction: Lower churn risk (negative class).")
        st.caption("The 0.50 decision threshold is a default classification threshold, not a business-approved intervention threshold.")

with tab4:
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"**Selected model:** {metrics['selected_model']}")
        cols = st.columns(5)
        for col, key, label in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                                   ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
            col.metric(label, f"{metrics[key]:.3f}")
        st.caption(f"Evaluation used {metrics['train_rows']:,} training rows and {metrics['test_rows']:,} held-out test rows.")
        st.caption("These are locally computed results after you run training. No performance value is assumed in advance.")
    comparison = ROOT / "reports" / "model_comparison.csv"
    if comparison.exists():
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(comparison), use_container_width=True, hide_index=True)
    image = ROOT / "reports" / "confusion_matrix.png"
    if image.exists():
        st.image(str(image), caption="Confusion matrix for selected model", use_container_width=False)

st.divider()
st.caption("Educational portfolio project. The IBM Telco sample represents a fictional provider; predictions should not be used for consequential decisions about real customers.")
