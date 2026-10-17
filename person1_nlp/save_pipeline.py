
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. FIND PROJECT AND DATASET
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent

POSSIBLE_DATA_DIRS = [
    SCRIPT_DIR / "data" / "processed",
    SCRIPT_DIR.parent / "data" / "processed",
]

DATA_PATH = next(
    (
        path for path in POSSIBLE_DATA_DIRS
        if (path / "gmrid_train_processed.csv").exists()
    ),
    None,
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "Training CSV not found. Searched:\n"
        + "\n".join(str(p) for p in POSSIBLE_DATA_DIRS)
    )

TRAIN_FILE = DATA_PATH / "gmrid_train_processed.csv"
MODEL_PATH = SCRIPT_DIR / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

print("Dataset:", TRAIN_FILE)
print("Model output:", MODEL_PATH)

# ==========================================
# 2. LOAD AND CLEAN DATA
# ==========================================

df = pd.read_csv(TRAIN_FILE)

required_columns = ["clean_text", "Summarized_label"]

for column in required_columns:
    if column not in df.columns:
        raise KeyError(f"Missing required column: {column}")

df = df.dropna(subset=["Summarized_label"]).copy()
df["clean_text"] = df["clean_text"].fillna("").astype(str)

df = df[df["clean_text"].str.strip().ne("")].copy()

X_train = df["clean_text"]
y_train = df["Summarized_label"]

print("\nTraining records:", len(df))
print("Number of classes:", y_train.nunique())

# ==========================================
# 3. BUILD COMPLETE ML PIPELINE
# ==========================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

# ==========================================
# 4. TRAIN PIPELINE
# ==========================================

print("\nTraining Logistic Regression pipeline...")
pipeline.fit(X_train, y_train)

# ==========================================
# 5. SAVE PIPELINE
# ==========================================

output_file = MODEL_PATH / "supply_chain_event_pipeline.joblib"

joblib.dump(pipeline, output_file)

print("\nPipeline saved successfully:")
print(output_file.resolve())

print("\nLearned classes:")
print(pipeline.named_steps["classifier"].classes_)

# ==========================================
# 6. TEST SAMPLE PREDICTION
# ==========================================

sample_text = [
    "Heavy rainfall caused flooding and delayed transportation."
]

prediction = pipeline.predict(sample_text)

print("\nSample text:", sample_text[0])
print("Predicted event category:", prediction[0])

print("\nPipeline setup completed.")
