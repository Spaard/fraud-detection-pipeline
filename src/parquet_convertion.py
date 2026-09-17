import pandas as pd

# Load raw CSV and export to compressed parquet
df = pd.read_csv("data/test.csv")
df.to_parquet("data/test.parquet", index=False)