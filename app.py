import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.data_loader import (
    load_transaction_data,
    prepare_map_data,
    train_fraud_model,
)

st.set_page_config(
    page_title="Card Payment Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
)

sns.set_theme(style="whitegrid")


@st.cache_data
def get_data():
    return load_transaction_data("data/test.parquet")


@st.cache_resource
def get_model(_df):
    return train_fraud_model(_df)


df = get_data()

st.title("🛡️ Card Payment Fraud Analytics & ML Alert System")
st.markdown(
    "Interactive MLOps dashboard for monitoring transaction fraud risk, exploring geographic patterns, and tuning alert thresholds."
)

st.markdown("---")

# 1. Business Metrics
st.subheader("1. Financial Loss & Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

total_tx = len(df)
total_frauds = int(df["is_fraud"].sum())
fraud_rate = (total_frauds / total_tx * 100) if total_tx > 0 else 0.0
total_fraud_val = df[df["is_fraud"] == 1]["movement_amount_euros"].sum()
median_legit = df[df["is_fraud"] == 0]["movement_amount_euros"].median()
median_fraud = df[df["is_fraud"] == 1]["movement_amount_euros"].median()

col1.metric("Total Transactions", f"{total_tx:,}")
col2.metric("Confirmed Frauds", f"{total_frauds:,}")
col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")
col4.metric("Total Loss Value", f"€{total_fraud_val:,.2f}")

st.info(
    f"**Business Insight**: The median fraudulent transaction amount (€{median_fraud:.2f}) is "
    f"**{median_fraud/median_legit:.1f}x higher** than legitimate payments (€{median_legit:.2f})."
)

# 2. Interactive Map
st.markdown("---")
st.subheader("2. Geographic Risk Map")
st.markdown(
    "Hover over country markers to inspect detailed fraud counts and transaction amounts."
)

map_data = prepare_map_data(df)

if not map_data.empty:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["longitude", "latitude"],
        get_color="[231, 76, 60, 180]",
        get_radius="radius",
        radius_scale=1,
        radius_min_pixels=12,
        radius_max_pixels=45,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=46.0,
        longitude=2.0,
        zoom=2,
        pitch=0,
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Country Code:</b> {counterparty_country_code}<br/>"
                "<b>Fraud Count:</b> {fraud_count}<br/>"
                "<b>Total Loss:</b> {total_loss_str}<br/>"
                "<b>Mean Amount:</b> {mean_amount_str}<br/>"
                "<b>Min Amount:</b> {min_amount_str}<br/>"
                "<b>Max Amount:</b> {max_amount_str}",
                "style": {
                    "backgroundColor": "#1e293b",
                    "color": "white",
                    "fontSize": "13px",
                },
            },
        )
    )

# 3. Exploratory Data Analytics
st.markdown("---")
st.subheader("3. Fraud Pattern Analytics")

tab1, tab2, tab3 = st.columns(3)

with tab1:
    st.markdown("**Amount Distribution (Log Scale)**")
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    sns.boxplot(
        data=df,
        x="is_fraud",
        y="movement_amount_euros",
        hue="is_fraud",
        palette={0: "#2ecc71", 1: "#e74c3c"},
        legend=False,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_xlabel("")
    ax.set_ylabel("Amount (€)")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.markdown("**Frauds by Member Role**")
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    sns.countplot(
        data=df[df["is_fraud"] == 1],
        x="member_role",
        hue="member_role",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.markdown("**Frauds by Payment Method**")
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    sns.countplot(
        data=df[df["is_fraud"] == 1],
        x="payment_method",
        hue="payment_method",
        palette="magma",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    plt.xticks(rotation=25)
    plt.tight_layout()
    st.pyplot(fig)

# 4. ML Alert Simulator
st.markdown("---")
st.subheader("4. Machine Learning Alert Simulator")
st.markdown(
    "Adjust decision probability threshold to satisfy the constraint of catching **at least 30% of frauds (Recall $\\ge 0.30$)**."
)

pipeline, X_test, y_test = get_model(df)
y_probs = pipeline.predict_proba(X_test)[:, 1]

threshold = st.slider(
    "Probability Decision Threshold",
    min_value=0.01,
    max_value=0.90,
    value=0.50,
    step=0.01,
)

y_pred = (y_probs >= threshold).astype(int)

rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

m_col1, m_col2, m_col3 = st.columns(3)

m_col1.metric("Model Recall", f"{rec * 100:.1f}%")
m_col2.metric("Model Precision", f"{prec * 100:.1f}%")

if rec >= 0.30:
    m_col3.success("✅ Target Met: Recall $\\ge 30\\%$")
else:
    m_col3.error("❌ Target Missed: Recall < 30%")

cm_col1, cm_col2, cm_col3 = st.columns([1, 1.2, 1])

with cm_col2:
    st.markdown("<h5 style='text-align: center;'>Confusion Matrix</h5>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Clear", "Alert"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
        cbar=False,
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    st.pyplot(fig)

# 5. Inspection Table
st.markdown("---")
st.subheader("5. Suspicious Transaction Inspector")

results_df = X_test.copy()
results_df["Fraud_Probability"] = y_probs
results_df["Actual_Fraud"] = y_test
results_df["Flagged_Alert"] = y_pred

top_alerts = results_df.sort_values(
    by="Fraud_Probability", ascending=False
).head(20)

st.dataframe(
    top_alerts.style.format(
        {
            "movement_amount_euros": "€{:.2f}",
            "Fraud_Probability": "{:.2%}",
        }
    ),
    use_container_width=True,
)

# 6. Executive Report & Project Synthesis
st.markdown("---")
st.subheader("6. Executive Report & Methodology")

with st.expander("📌 Stakes Analysis"):
    st.markdown(
        """
    - **Financial Risk**: Fraud losses directly impact bank profitability, especially in B2B contexts where transaction values are high.
    - **Customer Friction**: Overly aggressive blocking rules freeze critical business operations and increase customer churn.
    - **Operational Workload**: Alert fatigue places a heavy manual review burden on fraud analysts. The optimal system balances high recall with manageable false positive rates.
    """
    )

with st.expander("⚠️ Model Limitations"):
    st.markdown(
        """
    - **Severe Class Imbalance**: With ~0.24% positive class presence, standard accuracy is misleading, requiring probability threshold tuning.
    - **Feature Static Nature**: Absence of temporal velocity features (e.g., transaction frequency in 1 hour) limits fraud ring detection.
    - **Concept Drift**: Fraud patterns evolve quickly; models require continuous retraining pipelines and real-time monitoring.
    """
    )

with st.expander("🛠️ Mandatory MLOps Lifecycle Steps"):
    st.markdown(
        """
    1. **Business Problem Framing**: Aligning metrics with business cost (e.g., prioritizing Recall over Accuracy).
    2. **Exploratory Data Analysis & Feature Engineering**: Uncovering distributions and encoding categorical variables.
    3. **Baseline Modeling & Validation**: Building interpretable models (Logistic Regression) with cross-validation.
    4. **Containerization & Testing**: Securing dependencies via `uv` lockfiles, Pytest suites, and Docker multi-stage builds.
    5. **CI/CD & Deployment**: Automating test execution and container delivery via GitHub Actions.
    6. **Monitoring & Drift Detection**: Tracking model degradation and false positive rates post-deployment.
    """
    )