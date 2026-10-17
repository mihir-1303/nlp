
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==================================================
# 1. LOCATE DATASETS AUTOMATICALLY
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Supports the script being in src/ or the project root.
PROJECT_DIR = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name == "src"
    else SCRIPT_DIR
)

POSSIBLE_DATA_DIRS = [
    PROJECT_DIR / "data" / "processed",
    PROJECT_DIR.parent / "data" / "processed",
]

DATA_DIR = next(
    (
        folder for folder in POSSIBLE_DATA_DIRS
        if (folder / "gmrid_train_processed.csv").exists()
        and (folder / "gmrid_test_processed.csv").exists()
    ),
    None,
)

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate both processed CSV files.\n"
        "Expected gmrid_train_processed.csv and "
        "gmrid_test_processed.csv under data/processed."
    )

TRAIN_FILE = DATA_DIR / "gmrid_train_processed.csv"
TEST_FILE = DATA_DIR / "gmrid_test_processed.csv"

print("=" * 60)
print("SUPPLY CHAIN EVENT CLASSIFIER")
print("LOGISTIC REGRESSION vs LINEAR SVM")
print("=" * 60)
print("Dataset directory:", DATA_DIR)

# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

TEXT = "clean_text"
TARGET = "Summarized_label"

for column in [TEXT, TARGET]:
    if column not in train_df.columns:
        raise KeyError(f"Missing training column: {column}")
    if column not in test_df.columns:
        raise KeyError(f"Missing testing column: {column}")

train_df = train_df.dropna(subset=[TARGET]).copy()
test_df = test_df.dropna(subset=[TARGET]).copy()

train_df[TEXT] = train_df[TEXT].fillna("").astype(str)
test_df[TEXT] = test_df[TEXT].fillna("").astype(str)

train_df = train_df[train_df[TEXT].str.strip().ne("")]
test_df = test_df[test_df[TEXT].str.strip().ne("")]

known_labels = set(train_df[TARGET].unique())
test_df = test_df[test_df[TARGET].isin(known_labels)].copy()

X_train = train_df[TEXT]
y_train = train_df[TARGET]
X_test = test_df[TEXT]
y_test = test_df[TARGET]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Classes:", y_train.nunique())

# ==================================================
# 3. CREATE TF-IDF FEATURES
# ==================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==================================================
# 4. TRAIN AND EVALUATE MODELS
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
    ),
}

results = []

for name, model in models.items():
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=model.classes_,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test, predictions, labels=model.classes_
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
    })

# ==================================================
# 5. SAVE COMPARISON RESULTS
# ==================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Macro_F1", ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))

results_file = DATA_DIR / "model_comparison_results.csv"
results_df.to_csv(results_file, index=False)

print("\nResults saved to:", results_file)
print("\nExperiment completed.")

