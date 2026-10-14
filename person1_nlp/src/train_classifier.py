
import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

base_path = Path(__file__).resolve().parent.parent

processed_path = base_path / "data" / "processed"
model_path = base_path / "models"

model_path.mkdir(parents=True, exist_ok=True)

train_file = processed_path / "gmrid_train_processed.csv"
test_file = processed_path / "gmrid_test_processed.csv"


# --------------------------------------------------
# 2. LOAD PROCESSED DATA
# --------------------------------------------------

print("Loading processed datasets...")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

target_column = "Summarized_label"

required_columns = ["clean_text", target_column]

for column in required_columns:
    if column not in train_df.columns:
        raise KeyError(f"Missing training column: {column}")

    if column not in test_df.columns:
        raise KeyError(f"Missing testing column: {column}")


# --------------------------------------------------
# 3. PREPARE TRAINING AND TESTING DATA
# --------------------------------------------------

# Remove records with empty text or missing labels
train_df = train_df.dropna(subset=[target_column]).copy()
test_df = test_df.dropna(subset=[target_column]).copy()

train_df["clean_text"] = train_df["clean_text"].fillna("")
test_df["clean_text"] = test_df["clean_text"].fillna("")

train_df = train_df[
    train_df["clean_text"].str.strip().ne("")
].copy()

test_df = test_df[
    test_df["clean_text"].str.strip().ne("")
].copy()

# Keep only labels available in the training dataset
known_labels = set(train_df[target_column].unique())

unseen_mask = ~test_df[target_column].isin(known_labels)
unseen_count = int(unseen_mask.sum())

if unseen_count:
    print(
        f"Warning: Excluding {unseen_count} test records "
        "whose labels are absent from training data."
    )
    test_df = test_df.loc[~unseen_mask].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "Training or testing data is empty after filtering."
    )

X_train = train_df["clean_text"]
y_train = train_df[target_column]

X_test = test_df["clean_text"]
y_test = test_df[target_column]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of classes:", y_train.nunique())
print("\nTraining label counts:")
print(y_train.value_counts())


# --------------------------------------------------
# 4. TF-IDF FEATURE EXTRACTION
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

# Never fit the vectorizer on the test dataset
X_test_tfidf = vectorizer.transform(X_test)

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# --------------------------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# --------------------------------------------------
# 6. EVALUATE MODEL
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )
)


# --------------------------------------------------
# 7. SAVE MODEL AND VECTORIZER
# --------------------------------------------------

model_file = model_path / "gmrid_classifier.joblib"
vectorizer_file = model_path / "tfidf_vectorizer.joblib"

joblib.dump(model, model_file)
joblib.dump(vectorizer, vectorizer_file)

print("\nSaved model:", model_file.resolve())
print("Saved vectorizer:", vectorizer_file.resolve())


# --------------------------------------------------
# 8. SAVE TEST PREDICTIONS
# --------------------------------------------------

results_df = test_df.copy()
results_df["predicted_label"] = y_pred
results_df["correct_prediction"] = (
    results_df[target_column] == results_df["predicted_label"]
)

predictions_file = (
    processed_path / "gmrid_test_predictions.csv"
)

results_df.to_csv(predictions_file, index=False)

print("Saved predictions:", predictions_file.resolve())

print("\nPerson 1 NLP classification step is complete!")
