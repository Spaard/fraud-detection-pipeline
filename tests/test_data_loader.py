import pandas as pd
import pytest
from src.data_loader import (
    calculate_fraud_metrics,
    filter_by_amount,
    load_transaction_data,
    prepare_map_data,
    train_fraud_model,
)


@pytest.fixture
def sample_data():
    """Provide a minimal transactions dataframe for testing."""
    return pd.DataFrame({
        "card_transaction_id": ["tx1", "tx2", "tx3", "tx4"],
        "time_settled": [
            "2022-01-01 10:00:00",
            "2022-01-01 11:00:00",
            "2022-01-01 12:00:00",
            "2022-01-01 13:00:00",
        ],
        "counterparty_country_code": ["FR", "IE", "US", "FR"],
        "movement_amount_euros": [10.0, 50.0, 150.0, 500.0],
        "is_fraud": [0, 0, 1, 1],
    })


def test_filter_by_amount_valid_range(sample_data):
    """Test filtering within a valid price range."""
    result = filter_by_amount(sample_data, 20.0, 200.0)
    assert len(result) == 2
    assert list(result["card_transaction_id"]) == ["tx2", "tx3"]


def test_filter_by_amount_empty_result(sample_data):
    """Test filtering when no records match criteria."""
    result = filter_by_amount(sample_data, 1000.0, 2000.0)
    assert len(result) == 0


def test_filter_by_amount_invalid_range(sample_data):
    """Verify ValueError when min_amount exceeds max_amount."""
    with pytest.raises(
        ValueError, match="min_amount cannot be greater than max_amount"
    ):
        filter_by_amount(sample_data, 100.0, 50.0)


def test_calculate_fraud_metrics_correct_values(sample_data):
    """Test fraud metrics calculation logic."""
    metrics = calculate_fraud_metrics(sample_data)
    assert metrics["total"] == 4
    assert metrics["fraud_count"] == 2
    assert metrics["fraud_rate"] == 0.5


def test_calculate_fraud_metrics_empty_df():
    """Test metrics computation on an empty dataframe to prevent zero division."""
    empty_df = pd.DataFrame(columns=["card_transaction_id", "is_fraud"])
    metrics = calculate_fraud_metrics(empty_df)
    assert metrics["total"] == 0
    assert metrics["fraud_rate"] == 0.0


def test_prepare_map_data_valid_coords(sample_data):
    """Test geographical aggregation for Pydeck mapping."""
    geo_df = prepare_map_data(sample_data)
    assert len(geo_df) == 2
    assert "latitude" in geo_df.columns
    assert "longitude" in geo_df.columns
    assert "min_amount" in geo_df.columns
    assert "max_amount" in geo_df.columns
    assert set(geo_df["counterparty_country_code"]) == {"US", "FR"}


def test_prepare_map_data_empty_df():
    """Verify map preparation returns an empty DataFrame if no fraud exists."""
    empty_df = pd.DataFrame(
        columns=[
            "is_fraud",
            "counterparty_country_code",
            "movement_amount_euros",
        ]
    )
    geo_df = prepare_map_data(empty_df)
    assert geo_df.empty


def test_load_transaction_data_file_not_found():
    """Verify Exception when loading a non-existent file path."""
    with pytest.raises(FileNotFoundError):
        load_transaction_data("invalid/path/file.parquet")


def test_load_transaction_data_datetime_parsing(tmp_path):
    """Verify 'time_settled' is converted to datetime type upon loading parquet."""
    parquet_file = tmp_path / "dummy.parquet"

    dummy_df = pd.DataFrame({
        "time_settled": ["2022-01-01 10:00:00"],
        "movement_amount_euros": [10.0],
        "is_fraud": [0],
    })
    dummy_df.to_parquet(parquet_file)

    df = load_transaction_data(str(parquet_file))
    assert pd.api.types.is_datetime64_any_dtype(df["time_settled"])


def test_train_fraud_model_execution():
    """Test machine learning pipeline training and output shapes."""
    dataset = pd.DataFrame({
        "movement_amount_euros": [10.0, 50.0, 100.0, 200.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
        "is_physical": [True, False, True, False, True, False, True, False, True, False],
        "member_role": ["owner", "admin", "owner", "admin", "owner", "admin", "owner", "admin", "owner", "admin"],
        "payment_method": ["online", "vpos", "online", "vpos", "online", "vpos", "online", "vpos", "online", "vpos"],
        "counterparty_country_code": ["FR", "IE", "FR", "IE", "FR", "IE", "FR", "IE", "FR", "IE"],
        "is_fraud": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })

    pipeline, X_test, y_test = train_fraud_model(dataset)

    assert pipeline is not None
    assert len(X_test) > 0
    assert len(y_test) > 0