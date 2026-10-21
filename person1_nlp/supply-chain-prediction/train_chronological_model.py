import joblib
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv(
    "DataCoSupplyChainDataset.csv",
    encoding="latin1",
    low_memory=False
)

# 2. Parse order dates
date_col = "order date (DateOrders)"
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col, "Late_delivery_risk"])

# 3. Use only information that could be known at order time
numeric_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Latitude",
    "Longitude"
]

categorical_features = [
    "Type",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Category Name",
    "Department Name",
    "Customer Segment"
]

target = "Late_delivery_risk"

# 4. Chronological split
train_df = df[df[date_col] < "2017-01-01"].copy()
test_df = df[df[date_col] >= "2017-01-01"].copy()

X_train = train_df[numeric_features + categorical_features]
y_train = train_df[target].astype(int)

X_test = test_df[numeric_features + categorical_features]
y_test = test_df[target].astype(int)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training date range:",
      train_df[date_col].min(), "to", train_df[date_col].max())
print("Testing date range:",
      test_df[date_col].min(), "to", test_df[date_col].max())

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).sort_index())

# 5. Preprocess features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 6. Build model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. Train using earlier orders
print("\nTraining model...")
model.fit(X_train, y_train)

# 8. Evaluate on later orders
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))


from sklearn.metrics import confusion_matrix

print("\nTHRESHOLD ERROR ANALYSIS")

for threshold in [0.5, 0.4, 0.3]:
    threshold_predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        threshold_predictions,
        labels=[0, 1]
    ).ravel()

    print(f"\nThreshold: {threshold}")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")
# Save the trained pipeline
output_path = Path("late_delivery_model.joblib")

joblib.dump(model, output_path)

print(f"\nModel saved to: {output_path.resolve()}")