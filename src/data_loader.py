import pandas as pd

def load_transaction_data(filepath: str = "data/test.parquet") -> pd.DataFrame:
    """Load and parse credit card transaction dataset from parquet format."""
    df = pd.read_parquet(filepath)
    if 'time_settled' in df.columns:
        df['time_settled'] = pd.to_datetime(df['time_settled'])
    return df