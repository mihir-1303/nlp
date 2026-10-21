print("Script started")

import pandas as pd

print("Pandas imported")

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1",
    low_memory=False
)

print("Dataset loaded:", df.shape)

date_col = "order date (DateOrders)"
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

print("Earliest order:", df[date_col].min())
print("Latest order:", df[date_col].max())

print("\nOrders by year:")
print(df[date_col].dt.year.value_counts().sort_index())