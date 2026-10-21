
import pandas as pd

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1",
    low_memory=False
)

print("Total rows:", len(df))
print("Unique Order IDs:", df["Order Id"].nunique())
print("Unique Order Item IDs:", df["Order Item Id"].nunique())

print("\nRows per order:")
print(df.groupby("Order Id").size().describe())

print("\nRepeated order-item IDs:", df["Order Item Id"].duplicated().sum())

print("\nTarget consistency within orders:")
risk_counts = df.groupby("Order Id")["Late_delivery_risk"].nunique()
print("Orders with multiple target labels:", (risk_counts > 1).sum())
