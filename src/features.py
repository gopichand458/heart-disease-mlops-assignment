"""Feature definitions and preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"
DATA_PATH = "data/processed/heart_disease_clean.csv"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple engineered features (personal touch, documented in report)."""
    df = df.copy()
    df["hr_reserve"] = (220 - df["age"]) - df["thalach"]
    return df


ENGINEERED = ["hr_reserve"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC + ENGINEERED),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def load_xy(path: str = DATA_PATH):
    df = add_features(pd.read_csv(path))
    return df.drop(columns=[TARGET]), df[TARGET]
