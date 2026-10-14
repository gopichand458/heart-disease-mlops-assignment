"""Clean the raw Heart Disease data and save a processed CSV.

- Binarise the target (0 = no disease, 1 = disease; raw values 1-4 -> 1)
- Report missing values (ca, thal have a few). Imputation is done later
  inside the sklearn Pipeline so it is learned on training data only.
- Cast categorical-coded columns to integer-friendly types.
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_disease_raw.csv")
PROCESSED_PATH = Path("data/processed/heart_disease_clean.csv")

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["target"] = (df["target"] > 0).astype(int)
    return df


def main() -> None:
    raw = load_raw()
    print("Raw shape:", raw.shape)
    print("Missing values per column:")
    print(raw.isna().sum()[raw.isna().sum() > 0])
    df = clean(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print("Clean shape:", df.shape)
    print("Class balance:")
    print(df["target"].value_counts().rename({0: "no disease", 1: "disease"}))
    print(f"Saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
