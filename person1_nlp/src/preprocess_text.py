
import pandas as pd
import re
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

# Project folder: person1_nlp
base_path = Path(__file__).resolve().parent.parent

raw_path = base_path / "data" / "raw"
processed_path = base_path / "data" / "processed"

# Create processed folder if it does not exist
processed_path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. DATASET FILE PATHS
# --------------------------------------------------

train_file = raw_path / "GMRID_v3-train.csv"
test_file = raw_path / "GMRID_v3-test.csv"

# Check whether dataset files exist
if not train_file.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {train_file}"
    )

if not test_file.exists():
    raise FileNotFoundError(
        f"Testing dataset not found: {test_file}"
    )


# --------------------------------------------------
# 3. LOAD DATASETS
# --------------------------------------------------

print("Loading GMRID datasets...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("\nOriginal dataset information")
print("-" * 50)
print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

print("\nTraining columns:")
print(train_df.columns.tolist())


# --------------------------------------------------
# 4. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "Headline",
    "Details",
    "id",
    "Severity",
    "Region",
    "Datetime",
    "Category",
    "Summarized_label"
]

for column in required_columns:
    if column not in train_df.columns:
        raise KeyError(
            f"Required column '{column}' is missing "
            "from the training dataset."
        )

    if column not in test_df.columns:
        raise KeyError(
            f"Required column '{column}' is missing "
            "from the testing dataset."
        )


# --------------------------------------------------
# 5. TEXT CLEANING FUNCTION
# --------------------------------------------------

def clean_text(text):
    """Clean a text value for NLP processing."""

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Replace non-letter characters with spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# 6. COMBINE HEADLINE AND DETAILS
# --------------------------------------------------

print("\nCombining headline and details...")

for df in [train_df, test_df]:

    # Replace missing text with empty strings
    headline = df["Headline"].fillna("").astype(str)
    details = df["Details"].fillna("").astype(str)

    # Combine both text fields
    df["text"] = headline + " " + details

    # Clean combined text
    df["clean_text"] = df["text"].apply(clean_text)


# --------------------------------------------------
# 7. SELECT IMPORTANT COLUMNS
# --------------------------------------------------

selected_columns = [
    "id",
    "clean_text",
    "Severity",
    "Region",
    "Datetime",
    "Category",
    "Summarized_label"
]

train_processed = train_df[selected_columns].copy()
test_processed = test_df[selected_columns].copy()


# --------------------------------------------------
# 8. HANDLE EMPTY TEXT
# --------------------------------------------------

train_empty_count = (
    train_processed["clean_text"].str.strip().eq("").sum()
)

test_empty_count = (
    test_processed["clean_text"].str.strip().eq("").sum()
)

print("\nEmpty cleaned text records:")
print("Training:", train_empty_count)
print("Testing:", test_empty_count)

# Keep records in the processed files for inspection.
# They can be filtered during model training.


# --------------------------------------------------
# 9. CHECK MISSING LABELS
# --------------------------------------------------

print("\nMissing Summarized_label values:")
print(
    "Training:",
    train_processed["Summarized_label"].isna().sum()
)

print(
    "Testing:",
    test_processed["Summarized_label"].isna().sum()
)


# --------------------------------------------------
# 10. SAVE PROCESSED DATASETS
# --------------------------------------------------

train_output = (
    processed_path / "gmrid_train_processed.csv"
)

test_output = (
    processed_path / "gmrid_test_processed.csv"
)

train_processed.to_csv(
    train_output,
    index=False
)

test_processed.to_csv(
    test_output,
    index=False
)


# --------------------------------------------------
# 11. DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nProcessed training shape:")
print(train_processed.shape)

print("\nProcessed testing shape:")
print(test_processed.shape)


# Display sample cleaned text
print("\nSample cleaned text:")

for index, text in enumerate(
    train_processed["clean_text"].head(3),
    start=1
):
    print(f"\nSample {index}:")
    print(text[:500])


# Display event category distribution
print("\nSummarized label distribution:")

print(
    train_processed["Summarized_label"]
    .value_counts(dropna=False)
)


# Display severity distribution
print("\nSeverity distribution:")

print(
    train_processed["Severity"]
    .value_counts(dropna=False)
)


# --------------------------------------------------
# 12. DISPLAY OUTPUT FILE PATHS
# --------------------------------------------------

print("\nProcessed files saved at:")

print(train_output.resolve())
print(test_output.resolve())

print("\nPerson 1 NLP preprocessing step is complete!")
