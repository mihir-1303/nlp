import pandas as pd
from pathlib import Path


# Get the project paths
base_path = Path(__file__).resolve().parent.parent
data_path = base_path / "data" / "raw"


files = [
    "sentiment.csv",
    "topics.csv",
    "countries.csv",
    "cities.csv",
    "unigrams_tf_idf.csv"
]


for file in files:

    file_path = data_path / file

    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    df = pd.read_csv(file_path)

    print("Rows and Columns:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())