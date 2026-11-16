"""Download the Credit Risk dataset from OpenML (id 43454) and save it as data/credit_risk.csv."""
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://openml.org/data/v1/download/22102279/Credit-Risk-Dataset.arff"
OUT = Path(__file__).resolve().parent / "data" / "credit_risk.csv"


def parse_arff(text: str) -> pd.DataFrame:
    cols, rows, in_data = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            cols.append(s.split()[1])
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append(s)
    return pd.read_csv(io.StringIO("\n".join(rows)), names=cols, na_values=["?"])


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        text = r.read().decode("utf-8")
    df = parse_arff(text)
    df.to_csv(OUT, index=False)
    print(f"saved {OUT} shape={df.shape}")


if __name__ == "__main__":
    sys.exit(main())
