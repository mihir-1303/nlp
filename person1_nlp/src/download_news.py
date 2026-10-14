import requests
import pandas as pd
from pathlib import Path
import time


query = '"supply chain disruption" OR "port closure" OR "shipping disruption" OR "cargo delay"'

url = "https://api.gdeltproject.org/api/v2/doc/doc"

params = {
    "query": query,
    "mode": "artlist",
    "format": "json",
    "maxrecords": 20,
    "timespan": "1d",
    "sort": "datedesc"
}

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("Collecting news from GDELT...")

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=30
)

print("Status Code:", response.status_code)

if response.status_code == 429:
    print("GDELT is temporarily rate-limiting the request.")
    print("Please wait for a few minutes and try again.")
    exit()

response.raise_for_status()

data = response.json()

articles = data.get("articles", [])

print("Number of articles:", len(articles))

if len(articles) == 0:
    print("No articles found.")
    exit()

df = pd.DataFrame(articles)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 articles:")
print(df.head())

output_folder = Path(__file__).resolve().parent.parent / "data" / "raw"
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "news.csv"

df.to_csv(output_file, index=False)

print("\nDataset saved successfully:")
print(output_file)