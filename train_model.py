
import json
import os
import platform
import mlflow
import mlflow.sklearn
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Define paths relative to this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR, "sms+spam+collection", "SMSSpamCollection"
)
MODEL_PATH = os.path.join(BASE_DIR, "spam.model.joblib")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")

# 2. Make experiment settings explicit.
RANDOM_SEED = 42
TEST_SIZE = 0.20

# 3. Load and validate the dataset.
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    names=["label", "message"],
)

df = df.dropna(subset=["label", "message"])
df = df.drop_duplicates(subset=["message"])
df = df[df["label"].isin(["ham", "spam"])].copy()

if df.empty or df["label"].nunique() != 2:
    raise ValueError("Dataset must contain ham and spam messages.")

X = df["message"].astype(str)
y = df["label"].map({"ham": 0, "spam": 1})

print(f"Dataset size: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())

# 4. Stratify to preserve class proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)

# 5. Build and train the existing model pipeline.
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB()),
])

model.fit(X_train, y_train)

# 6. Evaluate on the held-out test set.
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["ham", "spam"],
    output_dict=True,
    zero_division=0,
)
matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["ham", "spam"],
    zero_division=0,
))
print("\nConfusion matrix (rows=true, columns=predicted):")
print(matrix)

# 7. Save the trained model and experiment metrics.
os.makedirs(ARTIFACT_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)

metrics = {
    "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "random_seed": RANDOM_SEED,
    "test_size": TEST_SIZE,
    "dataset_rows_after_cleaning": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "class_distribution": {
        str(k): int(v)
        for k, v in df["label"].value_counts().items()
    },
    "accuracy": float(accuracy),
    "spam_precision": float(report["spam"]["precision"]),
    "spam_recall": float(report["spam"]["recall"]),
    "spam_f1_score": float(report["spam"]["f1-score"]),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "versions": {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"\nModel saved: {MODEL_PATH}")
print(f"Metrics saved: {METRICS_PATH}")


# MLflow experiment tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("AI-Spam-Detection")

with mlflow.start_run(run_name="spam-detection-baseline-2"):
    mlflow.log_params({
        "model_type": "MultinomialNB",
        "vectorizer": "TfidfVectorizer",
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    })

    mlflow.log_metrics({
        "accuracy": float(accuracy),
        "spam_precision": float(report["spam"]["precision"]),
        "spam_recall": float(report["spam"]["recall"]),
        "spam_f1_score": float(report["spam"]["f1-score"]),
    })

    mlflow.log_artifact(METRICS_PATH)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="spam_model",
    )

    print("MLflow run ID:", mlflow.active_run().info.run_id)

print("MLflow tracking completed.")
