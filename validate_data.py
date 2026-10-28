
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sms+spam+collection" / "SMSSpamCollection"
OUTPUT_PATH = BASE_DIR / "artifacts" / "data_validation.json"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["label", "message"],
    )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df[["label", "message"]].isna().any().any():
        raise ValueError("Dataset contains missing labels or messages.")

    invalid_labels = set(df["label"].unique()) - {"ham", "spam"}
    if invalid_labels:
        raise ValueError(f"Unexpected labels: {invalid_labels}")

    if df["label"].nunique() != 2:
          raise ValueError("Dataset must contain both ham and spam.")

    report = {
        "valid": True,
        "rows": int(len(df)),
        "labels": {
            str(k): int(v)
            for k, v in df["label"].value_counts().items()
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("Data validation passed.")

if __name__ == "__main__":
    main()
