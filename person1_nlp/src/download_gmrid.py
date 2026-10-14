import requests
from pathlib import Path


# Get project path
base_path = Path(__file__).resolve().parent.parent

# Raw data folder
raw_path = base_path / "data" / "raw"

# Create folder if it does not exist
raw_path.mkdir(parents=True, exist_ok=True)


# GMRID dataset files
files = {
    "GMRID_v3-train.csv":
        "https://huggingface.co/datasets/windchimeran/GMRID/resolve/main/GMRID_v3-train.csv",

    "GMRID_v3-test.csv":
        "https://huggingface.co/datasets/windchimeran/GMRID/resolve/main/GMRID_v3-test.csv"
}


for filename, url in files.items():

    print("\n" + "=" * 60)
    print("Downloading:", filename)
    print("=" * 60)

    try:
        response = requests.get(url, timeout=60)

        if response.status_code == 200:

            file_path = raw_path / filename

            with open(file_path, "wb") as file:
                file.write(response.content)

            print("Download successful!")
            print("Saved to:", file_path)

        else:
            print("Download failed!")
            print("Status code:", response.status_code)

    except Exception as e:
        print("Error:", e)


print("\nFinished.")