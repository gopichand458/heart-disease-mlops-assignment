"""Download the UCI Heart Disease (Cleveland) dataset and save it as CSV.

Uses the macOS/system trust store (via truststore) so it also works behind
corporate proxies. If a local copy of the zip exists at data/raw/heart_disease.zip
it is used instead of downloading.
"""
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
MEMBER = "processed.cleveland.data"
LOCAL_ZIP = Path("data/raw/heart_disease.zip")
RAW_PATH = Path("data/raw/heart_disease_raw.csv")
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def _ssl_context() -> ssl.SSLContext:
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def _get_zip_bytes() -> bytes:
    if LOCAL_ZIP.exists():
        print(f"Using local zip: {LOCAL_ZIP}")
        return LOCAL_ZIP.read_bytes()
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, context=_ssl_context(), timeout=60) as resp:
        return resp.read()


def download(path: Path = RAW_PATH) -> Path:
    with zipfile.ZipFile(io.BytesIO(_get_zip_bytes())) as zf:
        with zf.open(MEMBER) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS, na_values="?")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns to {path}")
    return path


if __name__ == "__main__":
    download()
