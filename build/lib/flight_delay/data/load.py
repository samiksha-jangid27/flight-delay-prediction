from pathlib import Path

import pandas as pd


def load_flights(path: str | Path) -> pd.DataFrame:
    """Load the raw flight dataset from CSV."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path.suffix}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df