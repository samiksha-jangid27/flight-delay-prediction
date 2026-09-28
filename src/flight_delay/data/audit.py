from pathlib import Path

import pandas as pd


def audit_dataset(path: str | Path) -> None:
    """Print a structured audit of the raw flight dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print("=" * 70)
    print("FLIGHT DATASET AUDIT")
    print("=" * 70)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:02d}. {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nUnique values:")
    for column in df.columns:
        print(f"{column}: {df[column].nunique(dropna=False)}")

    print("\nFirst 5 rows:")
    print(df.head().to_string())


if __name__ == "__main__":
    audit_dataset("data/raw/flights.csv")