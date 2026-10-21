
import pandas as pd

df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1",
    low_memory=False
)

print("Late delivery risk counts:")
print(df["Late_delivery_risk"].value_counts().sort_index())

print("\nLate delivery risk percentages:")
print(
    (df["Late_delivery_risk"].value_counts(normalize=True)
       .sort_index() * 100).round(2)
)

print("\nDelivery status counts:")
print(df["Delivery Status"].value_counts())

print("\nDate examples:")
print(df[[
    "order date (DateOrders)",
    "shipping date (DateOrders)"
]].head())
