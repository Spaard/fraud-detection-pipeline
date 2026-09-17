import pandas as pd
import pytest
from src.data_loader import (
    calculate_fraud_metrics,
    filter_by_amount,
    load_transaction_data,
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
        "movement_amount_euros": [10.0, 50.0, 150.0, 500.0],
        "is_fraud": [0, 0, 1, 0],
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
    assert metrics["fraud_count"] == 1
    assert metrics["fraud_rate"] == 0.25


def test_calculate_fraud_metrics_empty_df():
    """Test metrics computation on an empty dataframe to prevent zero division."""
    empty_df = pd.DataFrame(columns=["card_transaction_id", "is_fraud"])
    metrics = calculate_fraud_metrics(empty_df)
    assert metrics["total"] == 0
    assert metrics["fraud_rate"] == 0.0


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