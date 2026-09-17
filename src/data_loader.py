import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

COUNTRY_COORDS = {
    "FR": (46.2276, 2.2137),
    "IE": (53.4129, -8.2439),
    "IT": (41.8719, 12.5674),
    "US": (37.0902, -95.7129),
    "DE": (51.1657, 10.4515),
    "GB": (55.3781, -3.4360),
    "ES": (40.4637, -3.7492),
    "NL": (52.1326, 5.2913),
    "VN": (14.0583, 108.2772),
    "LU": (49.8153, 6.1296),
    "BE": (50.5039, 4.4699),
    "AE": (23.4241, 53.8478),
    "SG": (1.3521, 103.8198),
    "HK": (22.3193, 114.1694),
    "CA": (56.1304, -106.3468),
}


def load_transaction_data(filepath: str = "data/test.parquet") -> pd.DataFrame:
    """Load transaction dataset from Parquet file and parse timestamps."""
    df = pd.read_parquet(filepath)
    if "time_settled" in df.columns:
        df["time_settled"] = pd.to_datetime(df["time_settled"])
    return df


def filter_by_amount(
    df: pd.DataFrame, min_amount: float, max_amount: float
) -> pd.DataFrame:
    """Filter records within a specific EUR amount range."""
    if min_amount > max_amount:
        raise ValueError("min_amount cannot be greater than max_amount")
    return df[
        (df["movement_amount_euros"] >= min_amount)
        & (df["movement_amount_euros"] <= max_amount)
    ]


def calculate_fraud_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate fraud metrics for summary displays."""
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


def prepare_map_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fraud metrics and pre-format currency strings for Pydeck tooltips."""
    fraud_df = df[df["is_fraud"] == 1]
    if fraud_df.empty:
        return pd.DataFrame()

    geo_stats = (
        fraud_df.groupby("counterparty_country_code")
        .agg(
            fraud_count=("is_fraud", "count"),
            total_fraud_amount=("movement_amount_euros", "sum"),
            mean_amount=("movement_amount_euros", "mean"),
            min_amount=("movement_amount_euros", "min"),
            max_amount=("movement_amount_euros", "max"),
        )
        .reset_index()
    )

    geo_stats["latitude"] = geo_stats["counterparty_country_code"].map(
        lambda c: COUNTRY_COORDS.get(c, (np.nan, np.nan))[0]
    )
    geo_stats["longitude"] = geo_stats["counterparty_country_code"].map(
        lambda c: COUNTRY_COORDS.get(c, (np.nan, np.nan))[1]
    )

    # Pre-format formatted string fields for clean display in Pydeck HTML
    geo_stats["total_loss_str"] = geo_stats["total_fraud_amount"].apply(
        lambda x: f"€{x:,.2f}"
    )
    geo_stats["mean_amount_str"] = geo_stats["mean_amount"].apply(
        lambda x: f"€{x:,.2f}"
    )
    geo_stats["min_amount_str"] = geo_stats["min_amount"].apply(
        lambda x: f"€{x:,.2f}"
    )
    geo_stats["max_amount_str"] = geo_stats["max_amount"].apply(
        lambda x: f"€{x:,.2f}"
    )

    # Circle radius scaling
    geo_stats["radius"] = geo_stats["fraud_count"] * 15000

    return geo_stats.dropna(subset=["latitude", "longitude"])


def train_fraud_model(df: pd.DataFrame):
    """Train a balanced Logistic Regression pipeline for fraud scoring."""
    features = [
        "movement_amount_euros",
        "is_physical",
        "member_role",
        "payment_method",
        "counterparty_country_code",
    ]
    target = "is_fraud"

    data = df[features + [target]].dropna()

    X = data[features]
    y = data[target]

    num_cols = ["movement_amount_euros"]
    cat_cols = ["member_role", "payment_method", "counterparty_country_code"]
    bool_cols = ["is_physical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("bool", "passthrough", bool_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test