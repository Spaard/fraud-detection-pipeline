import pandas as pd


def load_transaction_data(filepath: str = "data/test.parquet") -> pd.DataFrame:
    """Load and parse credit card transaction dataset from parquet format."""
    df = pd.read_parquet(filepath)
    if "time_settled" in df.columns:
        df["time_settled"] = pd.to_datetime(df["time_settled"])
    return df


def filter_by_amount(
    df: pd.DataFrame, min_amount: float, max_amount: float
) -> pd.DataFrame:
    """Filter transactions within a specific EUR amount range."""
    if min_amount > max_amount:
        raise ValueError("min_amount cannot be greater than max_amount")
    return df[
        (df["movement_amount_euros"] >= min_amount)
        & (df["movement_amount_euros"] <= max_amount)
    ]


def calculate_fraud_metrics(df: pd.DataFrame) -> dict:
    """Compute key fraud indicators from transaction data."""
    total_transactions = len(df)
    if total_transactions == 0:
        return {"total": 0, "fraud_count": 0, "fraud_rate": 0.0}

    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = float(fraud_count / total_transactions)
    return {
        "total": total_transactions,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
    }