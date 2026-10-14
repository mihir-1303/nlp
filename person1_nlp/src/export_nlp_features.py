
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

base_path = Path(__file__).resolve().parent.parent

processed_path = base_path / "data" / "processed"
model_path = base_path / "models"

train_file = processed_path / "gmrid_train_processed.csv"
test_file = processed_path / "gmrid_test_processed.csv"

classifier_file = model_path / "gmrid_classifier.joblib"
vectorizer_file = model_path / "tfidf_vectorizer.joblib"

# Output files for Person 2
train_output = processed_path / "nlp_events_train.csv"
test_output = processed_path / "nlp_events_test.csv"


# --------------------------------------------------
# 2. CHECK REQUIRED FILES
# --------------------------------------------------

required_files = [
    train_file,
    test_file,
    classifier_file,
    vectorizer_file
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# --------------------------------------------------
# 3. LOAD MODEL, VECTORIZER, AND DATA
# --------------------------------------------------

print("Loading trained NLP model...")

model = joblib.load(classifier_file)
vectorizer = joblib.load(vectorizer_file)

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# --------------------------------------------------
# 4. PREDICTION AND FEATURE EXPORT FUNCTION
# --------------------------------------------------

def generate_nlp_features(df, dataset_name):
    """Generate structured event predictions from cleaned text."""

    df = df.copy()

    # Handle missing text
    df["clean_text"] = df["clean_text"].fillna("").astype(str)

    # Keep original rows, but mark records with no usable text
    valid_mask = df["clean_text"].str.strip().ne("")

    df["predicted_event_type"] = "Unknown"
    df["event_prediction_confidence"] = float("nan")

    if valid_mask.any():
        valid_text = df.loc[valid_mask, "clean_text"]

        # Transform using the fitted vectorizer
        text_features = vectorizer.transform(valid_text)

        # Predict event category
        predictions = model.predict(text_features)

        df.loc[valid_mask, "predicted_event_type"] = predictions

        # Logistic Regression supports predict_proba.
        # This is model confidence, not disruption probability.
        probabilities = model.predict_proba(text_features)
        confidence = probabilities.max(axis=1)

        df.loc[
            valid_mask,
            "event_prediction_confidence"
        ] = confidence

    # Keep metadata and event features useful to Person 2
    available_columns = [
        "id",
        "Datetime",
        "Region",
        "Category",
        "Severity",
        "Summarized_label",
        "clean_text",
        "predicted_event_type",
        "event_prediction_confidence"
    ]

    output_df = df[
        [column for column in available_columns if column in df.columns]
    ].copy()

    # Explicitly record which dataset the rows came from
    output_df["dataset_split"] = dataset_name

    return output_df


# --------------------------------------------------
# 5. GENERATE FEATURES
# --------------------------------------------------

print("\nGenerating training event features...")

train_features = generate_nlp_features(
    train_df,
    "train"
)

print("Generating testing event features...")

test_features = generate_nlp_features(
    test_df,
    "test"
)


# --------------------------------------------------
# 6. SAVE OUTPUT FILES
# --------------------------------------------------

train_features.to_csv(train_output, index=False)
test_features.to_csv(test_output, index=False)

print("\n" + "=" * 55)
print("NLP FEATURE EXPORT COMPLETED")
print("=" * 55)

print("\nTraining output shape:", train_features.shape)
print("Testing output shape:", test_features.shape)

print("\nPredicted event distribution (training):")
print(train_features["predicted_event_type"].value_counts())

print("\nPredicted event distribution (testing):")
print(test_features["predicted_event_type"].value_counts())

print("\nTraining file saved to:")
print(train_output.resolve())

print("\nTesting file saved to:")
print(test_output.resolve())

print("\nSample exported records:")
print(
    test_features[
        [
            "predicted_event_type",
            "event_prediction_confidence",
            "Severity",
            "Region"
        ]
    ].head()
)

print("\nPerson 1 NLP feature export is complete!")
