import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sms+spam+collection" / "SMSSpamCollection"
OUTPUT_PATH = BASE_DIR / "artifacts" / "prepared_data.csv"

def main():
    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["label", "message"],
    )

    #remove duplicates and missing values
    df = df.dropna(subset=["label", "message"])
    df = df.drop_duplicates()

    #remove empty messages and normalize labels
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["message"] != ""]

    #keep only expected labels
    df = df[df["label"].isin(["ham", "spam"])]

    if df.empty:
        raise ValueError("No valid data after preparation.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH,index=False)

    print(f"Prepared rows: {len(df)}")
    print(f"labels distribution:\n{df['label'].value_counts()}")
    print(f"Prepared data saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()






