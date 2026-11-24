
from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "processed" / "telco_churn_clean.csv"
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "best_model_metrics.json"

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📉",
    layout="wide"
)

# --------------------------------------------------
# AUTOMATIC DATASET PREPARATION AND MODEL TRAINING
# --------------------------------------------------

@st.cache_resource
def prepare_artifacts():
    """
    Automatically download, clean, and train models
    when required artifacts are missing.
    """

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        from src.train import main as train_model

        with st.spinner(
            "Preparing dataset and training ML models. "
            "This may take a few minutes..."
        ):
            train_model()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset was not generated."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model was not generated."
        )

    return True


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

st.title("📉 Customer Churn Analysis & Prediction")

st.caption(
    "Telecommunications customer retention analytics "
    "using the IBM Telco sample dataset."
)

try:
    prepare_artifacts()
    df = load_data()
    model = load_model()

except Exception as error:
    st.error("Unable to prepare the application.")
    st.exception(error)
    st.info(
        "Please check the Streamlit deployment logs "
        "for the underlying error."
    )
    st.stop()


# --------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------

df["ChurnLabel"] = df["Churn"].map({
    0: "Retained",
    1: "Churned"
})

churn_rate = df["Churn"].mean() * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total customers", f"{len(df):,}")
c2.metric("Churn rate", f"{churn_rate:.1f}%")
c3.metric("Churned customers", f"{int(df['Churn'].sum()):,}")
c4.metric("Average tenure", f"{df['tenure'].mean():.1f} months")


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Customer insights",
    "Predict churn",
    "Model results"
])


# ==================================================
# TAB 1: OVERVIEW
# ==================================================

with tab1:

    st.subheader("Customer Retention Overview")

    left, right = st.columns(2)

    with left:

        fig = px.pie(
            df,
            names="ChurnLabel",
            hole=0.55,
            title="Customer retention vs churn"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        by_contract = (
            df.groupby("Contract", as_index=False)["Churn"]
            .mean()
        )

        by_contract["Churn rate (%)"] = (
            by_contract["Churn"] * 100
        )

        fig = px.bar(
            by_contract,
            x="Contract",
            y="Churn rate (%)",
            title="Churn rate by contract type",
            text_auto=".1f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "This dashboard describes the churn labels in the "
        "supplied sample. It does not establish that a "
        "particular business action causes churn to decrease."
    )


# ==================================================
# TAB 2: CUSTOMER INSIGHTS
# ==================================================

with tab2:

    st.subheader("Customer Behaviour Analysis")

    left, right = st.columns(2)

    with left:

        fig = px.histogram(
            df,
            x="tenure",
            color="ChurnLabel",
            barmode="overlay",
            nbins=24,
            title="Tenure distribution by churn status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig = px.box(
            df,
            x="ChurnLabel",
            y="MonthlyCharges",
            color="ChurnLabel",
            title="Monthly charges by churn status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Churn Rate by Payment Method")

    pay = (
        df.groupby("PaymentMethod", as_index=False)["Churn"]
        .mean()
    )

    pay["Churn rate (%)"] = pay["Churn"] * 100

    fig = px.bar(
        pay,
        x="PaymentMethod",
        y="Churn rate (%)",
        title="Churn rate by payment method",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# TAB 3: PREDICT CUSTOMER CHURN
# ==================================================

with tab3:

    st.subheader("Predict Churn for a Customer")

    with st.form("prediction_form"):

        a, b, c = st.columns(3)

        with a:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (months)",
                0, 72, 12
            )

            phone = st.selectbox(
                "Phone service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"]
            )

        with b:

            internet = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No"]
            )

            security = st.selectbox(
                "Online security",
                ["No", "Yes", "No internet service"]
            )

            backup = st.selectbox(
                "Online backup",
                ["No", "Yes", "No internet service"]
            )

            protection = st.selectbox(
                "Device protection",
                ["No", "Yes", "No internet service"]
            )

            support = st.selectbox(
                "Tech support",
                ["No", "Yes", "No internet service"]
            )

            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

            movies = st.selectbox(
                "Streaming movies",
                ["No", "Yes", "No internet service"]
            )

        with c:

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0
            )

            total = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=20000.0,
                value=float(monthly * tenure)
            )

        submitted = st.form_submit_button(
            "Predict churn",
            type="primary"
        )

    if submitted:

        record = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:

            probability = float(
                model.predict_proba(record)[0, 1]
            )

            prediction = int(probability >= 0.5)

            st.metric(
                "Estimated churn probability",
                f"{probability:.1%}"
            )

            if prediction:

                st.error(
                    "Model prediction: Higher churn risk "
                    "(positive class)."
                )

            else:

                st.success(
                    "Model prediction: Lower churn risk "
                    "(negative class)."
                )

            st.caption(
                "The 0.50 decision threshold is a default "
                "classification threshold, not a "
                "business-approved intervention threshold."
            )

        except Exception as error:

            st.error("Prediction failed.")
            st.exception(error)


# ==================================================
# TAB 4: MODEL RESULTS
# ==================================================

with tab4:

    st.subheader("Machine Learning Model Performance")

    if METRICS_PATH.exists():

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metrics = json.load(file)

        st.write(
            f"**Selected model:** {metrics['selected_model']}"
        )

        cols = st.columns(5)

        metric_items = [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1-score"),
            ("roc_auc", "ROC-AUC")
        ]

        for col, (key, label) in zip(cols, metric_items):

            col.metric(
                label,
                f"{metrics[key]:.3f}"
            )

        st.caption(
            f"Evaluation used {metrics['train_rows']:,} "
            f"training rows and {metrics['test_rows']:,} "
            "held-out test rows."
        )

        st.caption(
            "These are held-out evaluation results on the "
            "IBM sample dataset, not a guarantee of "
            "real-world performance."
        )

    comparison = ROOT / "reports" / "model_comparison.csv"

    if comparison.exists():

        st.subheader("Model Comparison")

        comparison_df = pd.read_csv(comparison)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    image = ROOT / "reports" / "confusion_matrix.png"

    if image.exists():

        st.subheader("Confusion Matrix")

        st.image(
            str(image),
            caption="Confusion matrix for selected model",
            use_container_width=False
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Educational portfolio project. The IBM Telco sample "
    "represents a fictional provider; predictions should "
    "not be used for consequential decisions about real customers."
)
