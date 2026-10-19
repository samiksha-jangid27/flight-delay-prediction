
import pandas as pd
import pytest

from flight_delay.data.audit import audit_dataset


def test_audit_dataset_prints_report(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "carrier": ["AA", "BB", "AA"],
            "arr_delay": [10.0, None, 10.0],
        }
    )

    path = tmp_path / "flights.csv"
    df.to_csv(path, index=False)

    audit_dataset(path)
    output = capsys.readouterr().out

    assert "FLIGHT DATASET AUDIT" in output
    assert "Shape: (3, 2)" in output
    assert "Data types:" in output
    assert "Missing values:" in output
    assert "Duplicate rows: 1" in output
    assert "Unique values:" in output
    assert "First 5 rows:" in output
    assert "carrier" in output
    assert "arr_delay" in output


def test_audit_dataset_missing_file_raises_error(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        audit_dataset(path)