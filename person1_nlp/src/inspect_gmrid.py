import pandas as pd
from pathlib import Path


# Get project path
base_path = Path(__file__).resolve().parent.parent

# Raw data folder
data_path = base_path / "data" / "raw"


files = [
    "GMRID_v3-train.csv",
    "GMRID_v3-test.csv"
]


for file in files:

    file_path = data_path / file

    print("\n" + "=" * 70)
    print("FILE:", file)
    print("=" * 70)

    # Read CSV
    df = pd.read_csv(file_path)

    print("\nRows and Columns:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nUnique Values:")
    for column in df.columns:
        if df[column].nunique() < 30:
            print(f"\n{column}:")
            print(df[column].unique())
            