
import pandas as pd

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1",
    low_memory=False
)

print("Late-delivery risk vs delivery status:")
print(pd.crosstab(
    df["Delivery Status"],
    df["Late_delivery_risk"],
    margins=True
))

print("\nLate-delivery risk vs scheduled shipping days:")
print(pd.crosstab(
    df["Days for shipment (scheduled)"],
    df["Late_delivery_risk"],
    normalize="index"
).round(3))

print("\nShipping modes:")
print(df["Shipping Mode"].value_counts())

print("\nOrder date range:")
dates = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)
print("Start:", dates.min())
print("End:", dates.max())
