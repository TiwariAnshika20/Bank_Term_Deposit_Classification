
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Analytics | Term Deposit AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

[data-testid="stSidebar"] {
    background-color: #14213D;
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #14213D;
}

.subtitle {
    color: #64748B;
    font-size: 15px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

.metric-label {
    color: #64748B;
    font-size: 13px;
}

.metric-value {
    color: #14213D;
    font-size: 28px;
    font-weight: 800;
}

.section-title {
    color: #14213D;
    font-size: 23px;
    font-weight: 700;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k-nearest_neighbors.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

@st.cache_resource
def load_models():

    loaded = {}

    for name, filename in MODEL_FILES.items():

        path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(path):
            loaded[name] = joblib.load(path)

    return loaded


models = load_models()


# --------------------------------------------------
# LOAD SAMPLE TEST DATA
# --------------------------------------------------

@st.cache_data
def load_sample_data():

    path = os.path.join(DATA_DIR, "test_data.csv")

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


sample_data = load_sample_data()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("# 🏦 BANK AI")
st.sidebar.markdown("### Analytics Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Model Performance",
        "Model Evaluation",
        "Customer Prediction",
        "Test Dataset"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Information")
st.sidebar.caption("Bank Term Deposit Classification")
st.sidebar.caption("Machine Learning Assignment 2")
st.sidebar.caption("Five Classification Algorithms")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def evaluate_models(data):

    if "y" not in data.columns:
        raise ValueError(
            "The uploaded CSV must contain the target column 'y'."
        )

    X = data.drop(columns=["y"])
    y_true = data["y"].astype(str).str.lower()

    if not set(y_true.unique()).issubset({"yes", "no"}):
        raise ValueError(
            "Target column 'y' must contain only yes/no values."
        )

    if set(y_true.unique()) != {"yes", "no"}:
        raise ValueError(
            "The uploaded dataset must contain both yes and no classes "
            "to calculate all evaluation metrics."
        )

    results = []
    predictions = {}

    for name, model in models.items():

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]
        else:
            y_score = model.decision_function(X)

        y_binary = (y_true == "yes").astype(int)

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_binary, y_score),
            "Precision": precision_score(
                y_true, y_pred, pos_label="yes", zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, pos_label="yes", zero_division=0
            ),
            "F1 Score": f1_score(
                y_true, y_pred, pos_label="yes", zero_division=0
            ),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }

        results.append(result)
        predictions[name] = y_pred

    return pd.DataFrame(results), predictions, y_true


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown(
        '<div class="main-title">Bank Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered customer term deposit classification</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("""
    This platform evaluates five machine learning algorithms
    to predict whether a bank customer is likely to subscribe
    to a term deposit.
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Dataset Records", "45,211")

    with c2:
        metric_card("Input Features", "15")

    with c3:
        metric_card("ML Models", "5")

    with c4:
        metric_card("Target Classes", "2")

    st.write("")

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Classification Objective")

        st.write(
            "Predict whether a customer will subscribe to a "
            "bank term deposit using customer information "
            "and marketing campaign features."
        )

        st.markdown("### Target Variable")

        st.info("y = yes / no")

    with col2:

        st.markdown("### Machine Learning Algorithms")

        for name in MODEL_FILES:
            st.write("• " + name)

    st.write("")

    st.markdown("### Evaluation Metrics")

    st.write(
        "Accuracy | AUC | Precision | Recall | F1 Score | MCC"
    )

    if not models:
        st.error("No trained models were found. Check the model folder.")


# --------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Comparison of all five trained algorithms</div>',
        unsafe_allow_html=True
    )

    comparison_path = os.path.join(
        BASE_DIR, "model_comparison.csv"
    )

    if os.path.exists(comparison_path):

        results = pd.read_csv(comparison_path)

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        metric = st.selectbox(
            "Select Evaluation Metric",
            [
                "Accuracy", "AUC", "Precision",
                "Recall", "F1 Score", "MCC"
            ]
        )

        fig = px.bar(
            results,
            x="Model",
            y=metric,
            color="Model",
            text_auto=".4f",
            title=f"{metric} Comparison"
        )

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            height=480
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Overall Metric Comparison")

        melted = results.melt(
            id_vars="Model",
            value_vars=[
                "Accuracy", "AUC", "Precision",
                "Recall", "F1 Score", "MCC"
            ],
            var_name="Metric",
            value_name="Score"
        )

        fig2 = px.bar(
            melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="All Evaluation Metrics"
        )

        fig2.update_layout(template="plotly_white")

        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Model comparison file not found.")


# --------------------------------------------------
# PAGE 3: MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.markdown(
        '<div class="main-title">Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Detailed evaluation on test data</div>',
        unsafe_allow_html=True
    )

    if sample_data is None:
        st.warning("Sample test dataset not found.")
    else:

        selected_model = st.selectbox(
            "Choose Classification Model",
            list(models.keys())
        )

        model = models[selected_model]

        X_eval = sample_data.drop(columns=["y"])
        y_true = sample_data["y"].astype(str).str.lower()

        y_pred = model.predict(X_eval)

        y_binary = (y_true == "yes").astype(int)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_eval)[:, 1]
        else:
            y_score = model.decision_function(X_eval)

        m1, m2, m3 = st.columns(3)

        with m1:
            metric_card(
                "Accuracy",
                f"{accuracy_score(y_true, y_pred):.4f}"
            )

        with m2:
            metric_card(
                "AUC",
                f"{roc_auc_score(y_binary, y_score):.4f}"
            )

        with m3:
            metric_card(
                "F1 Score",
                f"{f1_score(y_true, y_pred, pos_label='yes'):.4f}"
            )

        st.write("")

        cm = confusion_matrix(
            y_true, y_pred, labels=["no", "yes"]
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Predicted No", "Predicted Yes"],
            y=["Actual No", "Actual Yes"],
            color_continuous_scale="Blues",
            title=f"{selected_model} - Confusion Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            labels=["no", "yes"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )


# --------------------------------------------------
# PAGE 4: CUSTOMER PREDICTION
# --------------------------------------------------

elif page == "Customer Prediction":

    st.markdown(
        '<div class="main-title">Customer Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Predict subscription using customer details</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter customer details to receive a prediction "
        "from your selected trained model."
    )

    with st.form("customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age", min_value=18, max_value=100, value=35
            )

            job = st.selectbox(
                "Job",
                [
                    "admin.", "blue-collar", "entrepreneur",
                    "housemaid", "management", "retired",
                    "self-employed", "services", "student",
                    "technician", "unemployed", "unknown"
                ]
            )

            marital = st.selectbox(
                "Marital Status",
                ["married", "single", "divorced"]
            )

            education = st.selectbox(
                "Education",
                ["primary", "secondary", "tertiary", "unknown"]
            )

            default = st.selectbox(
                "Credit in Default", ["no", "yes", "unknown"]
            )

        with col2:

            balance = st.number_input(
                "Account Balance", value=1000
            )

            housing = st.selectbox(
                "Housing Loan", ["yes", "no", "unknown"]
            )

            loan = st.selectbox(
                "Personal Loan", ["no", "yes", "unknown"]
            )

            contact = st.selectbox(
                "Contact Type", ["cellular", "telephone", "unknown"]
            )

            day = st.number_input(
                "Day of Week (1–31)",
                min_value=1, max_value=31, value=15
            )

        with col3:

            month = st.selectbox(
                "Month",
                [
                    "jan", "feb", "mar", "apr", "may",
                    "jun", "jul", "aug", "sep",
                    "oct", "nov", "dec"
                ]
            )

            campaign = st.number_input(
                "Number of Contacts",
                min_value=1, value=1
            )

            pdays = st.number_input(
                "Days Since Previous Contact",
                min_value=-1, value=-1
            )

            previous = st.number_input(
                "Previous Contacts",
                min_value=0, value=0
            )

            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                ["unknown", "failure", "other", "success"]
            )

        selected_model = st.selectbox(
            "Select Model",
            list(models.keys()),
            key="prediction_model"
        )

        submit = st.form_submit_button(
            "Predict Subscription",
            use_container_width=True
        )

    if submit:

        input_data = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day_of_week": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        model = models[selected_model]

        prediction = model.predict(input_data)[0]

        st.write("")

        if prediction == "yes":
            st.success(
                "Prediction: Customer is likely to subscribe (YES)."
            )
        else:
            st.warning(
                "Prediction: Customer is unlikely to subscribe (NO)."
            )

        st.caption(
            "This is a machine learning prediction, not a guarantee "
            "of customer behavior."
        )


# --------------------------------------------------
# PAGE 5: TEST DATASET UPLOAD
# --------------------------------------------------

elif page == "Test Dataset":

    st.markdown(
        '<div class="main-title">Test Dataset Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Upload a labeled CSV and evaluate all five models</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Upload a CSV containing the 15 input features and "
        "the actual target column 'y' (yes/no)."
    )

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Dataset Preview")

            st.dataframe(
                uploaded_data.head(),
                use_container_width=True
            )

            if st.button("Evaluate All Models"):

                results, predictions, y_true = evaluate_models(
                    uploaded_data
                )

                st.markdown("### Model Comparison Results")

                st.dataframe(
                    results.style.format({
                        col: "{:.4f}"
                        for col in [
                            "Accuracy", "AUC", "Precision",
                            "Recall", "F1 Score", "MCC"
                        ]
                    }),
                    use_container_width=True
                )

                fig = px.bar(
                    results.melt(
                        id_vars="Model",
                        value_vars=[
                            "Accuracy", "AUC", "Precision",
                            "Recall", "F1 Score", "MCC"
                        ],
                        var_name="Metric",
                        value_name="Score"
                    ),
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Uploaded Dataset - Model Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                selected = st.selectbox(
                    "Select Model for Detailed Evaluation",
                    list(models.keys())
                )

                selected_pred = predictions[selected]

                cm = confusion_matrix(
                    y_true,
                    selected_pred,
                    labels=["no", "yes"]
                )

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=["Predicted No", "Predicted Yes"],
                    y=["Actual No", "Actual Yes"],
                    color_continuous_scale="Blues",
                    title=f"{selected} - Confusion Matrix"
                )

                st.plotly_chart(
                    fig_cm,
                    use_container_width=True
                )

                report = classification_report(
                    y_true,
                    selected_pred,
                    labels=["no", "yes"],
                    output_dict=True,
                    zero_division=0
                )

                st.markdown("### Classification Report")

                st.dataframe(
                    pd.DataFrame(report).transpose(),
                    use_container_width=True
                )

                st.success(
                    "Evaluation completed. Models were not retrained."
                )

        except Exception as error:
            st.error(f"Evaluation error: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748B;padding:10px;">
        Bank Term Deposit Classification |
        Machine Learning Assignment 2
    </div>
    """,
    unsafe_allow_html=True
)
