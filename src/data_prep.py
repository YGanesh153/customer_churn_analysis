from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED = ROOT / "data" / "processed" / "telco_churn_clean.csv"

def load_and_clean():
    if not RAW.exists():
        from src.download_data import main as download
        download()

    df = pd.read_csv(RAW)
    required = {"customerID", "Churn", "TotalCharges", "tenure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.drop_duplicates(subset=["customerID"]).copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Blank TotalCharges values occur for brand-new customers (tenure == 0).
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    print(f"Cleaned data saved: {PROCESSED}")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    return df

if __name__ == "__main__":
    load_and_clean()
