import streamlit as st
from src.data_loader import load_transaction_data, filter_by_amount

st.set_page_config(page_title="Fraud Analytics Dashboard", layout="wide")
st.title("Card Transaction Fraud Dashboard")

@st.cache_data
def get_data():
    return load_transaction_data("data/test.csv")

# Load data with cached execution
data_load_state = st.text("Loading dataset...")
df = get_data()
data_load_state.text("Dataset loaded successfully!")

# Sidebar filtering controls
st.sidebar.header("Filters")
min_val = float(df['movement_amount_euros'].min())
max_val = float(df['movement_amount_euros'].max())
selected_range = st.sidebar.slider("Amount (€)", min_val, max_val, (min_val, max_val))

filtered_df = filter_by_amount(df, selected_range[0], selected_range[1])

# Key performance indicators
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{len(filtered_df):,}")
col2.metric("Total Fraud Cases", int(filtered_df['is_fraud'].sum()))
col3.metric("Fraud Rate", f"{(filtered_df['is_fraud'].mean() * 100):.2f}%")

# Data visualizer
st.subheader("Transaction Volume by Fraud Status")
fraud_counts = filtered_df['is_fraud'].value_counts().rename({0: 'Legitimate', 1: 'Fraud'})
st.bar_chart(fraud_counts)

if st.checkbox("Show raw transaction data"):
    st.subheader("Raw Data Sample")
    st.dataframe(filtered_df.head(100))