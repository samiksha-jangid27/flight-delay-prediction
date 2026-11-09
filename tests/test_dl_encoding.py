"""Tests for the DL tensor encoder and categorical embedding tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from flight_delay.deep_learning.dataset import DLDataset, DLPartition
from flight_delay.deep_learning.embeddings import (
    EMBEDDING_FIELDS,
    build_embedding_tables,
    default_embedding_dimensions,
)
from flight_delay.deep_learning.encoding import DLTensorEncoder
from flight_delay.deep_learning.mlp import FlightDelayMLP


NUMERIC_FIELDS = ("num_a", "num_b")
CATEGORICAL_FIELDS = ("route", "dest", "carrier", "distance_band", "origin")


def _metadata(labels: list[int] | None = None) -> pd.DataFrame:
    if labels is None:
        labels = [0, 1, 0]
    return pd.DataFrame(
        {
            "flight_id": ["0001", "0002", "0003"][: len(labels)],
            "flight_date": pd.to_datetime(
                ["2013-01-01", "2013-01-02", "2013-01-03"][: len(labels)]
            ),
            "y_true": labels,
            "partition": ["train"] * len(labels),
        }
    )


def _vocabularies() -> dict[str, dict[str, int]]:
    return {
        "carrier": {"AA": 2, "DL": 3},
        "origin": {"JFK": 2, "LGA": 3},
        "dest": {"LAX": 2, "SFO": 3},
        "route": {"JFK-LAX": 2, "LGA-SFO": 3},
        "distance_band": {"short": 2, "long": 3},
    }


def _partition(
    *,
    numeric: np.ndarray | None = None,
    categorical: np.ndarray | None = None,
    labels: list[int] | None = None,
    metadata: pd.DataFrame | None = None,
) -> DLPartition:
    if numeric is None:
        numeric = np.array(
            [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]], dtype=np.float32
        )
    if categorical is None:
        # Source order is route, dest, carrier, distance_band, origin.
        categorical = np.array(
            [
                [2, 3, 2, 3, 2],  # known categories
                [0, 1, 3, 0, 1],  # missing / unseen categories
                [3, 2, 2, 2, 3],  # another known combination
            ],
            dtype=np.int64,
        )
    if labels is None:
        labels = [0, 1, 0]
    if metadata is None:
        metadata = _metadata(labels)
    return DLPartition(numeric=numeric, categorical=categorical, metadata=metadata)


def _dataset(partition: DLPartition | None = None) -> DLDataset:
    if partition is None:
        partition = _partition()
    vocabularies = _vocabularies()
    return DLDataset(
        partitions={"train": partition, "validation": partition},
        medians=pd.Series({"num_a": 0.0, "num_b": 0.0}),
        means=pd.Series({"num_a": 0.0, "num_b": 0.0}),
        scales=pd.Series({"num_a": 1.0, "num_b": 1.0}),
        vocabularies=vocabularies,
        numeric_features=NUMERIC_FIELDS,
        categorical_features=CATEGORICAL_FIELDS,
    )


def test_transform_returns_mlp_compatible_shapes_dtypes_and_order() -> None:
    partition = _partition()
    encoder = DLTensorEncoder(_dataset(partition))

    batch = encoder.transform(partition)

    # Dense: 2 numeric + 4 route columns + 4 distance-band columns.
    assert tuple(batch.x_dense.shape) == (3, 10)
    assert tuple(batch.x_cat.shape) == (3, 3)
    assert tuple(batch.labels.shape) == (3,)
    assert batch.x_dense.dtype == torch.float32
    assert batch.x_cat.dtype == torch.long
    assert batch.labels.dtype == torch.float32
    assert batch.metadata["flight_id"].tolist() == ["0001", "0002", "0003"]
    assert batch.x_cat.tolist() == [[2, 2, 3], [3, 1, 1], [2, 3, 2]]


def test_one_hot_encodes_known_missing_and_unseen_values() -> None:
    partition = _partition()
    batch = DLTensorEncoder(_dataset(partition)).transform(partition)
    # route occupies dense columns 2:6; distance_band occupies 6:10.
    assert batch.x_dense[0, 2:6].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert batch.x_dense[1, 2:6].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert batch.x_dense[1, 6:10].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert batch.x_dense[0, 6:10].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_transform_partition_uses_fitted_dataset_partition() -> None:
    dataset = _dataset()
    batch = DLTensorEncoder(dataset).transform_partition("validation")
    assert batch.size == 3


def test_dense_feature_names_match_dense_tensor_width() -> None:
    encoder = DLTensorEncoder(_dataset())
    assert len(encoder.dense_feature_names) == encoder.dense_dim == 10
    assert encoder.dense_feature_names[:2] == NUMERIC_FIELDS
    assert encoder.dense_feature_names[2] == "route__missing"
    assert encoder.dense_feature_names[3] == "route__unseen"


def test_transform_preserves_metadata_without_aliasing_it() -> None:
    partition = _partition()
    batch = DLTensorEncoder(_dataset(partition)).transform(partition)
    batch.metadata.loc[0, "flight_id"] = "changed"
    assert partition.metadata.loc[0, "flight_id"] == "0001"


def test_unknown_partition_name_raises() -> None:
    encoder = DLTensorEncoder(_dataset())
    with pytest.raises(ValueError, match="Unknown partition"):
        encoder.transform_partition("test")


def test_nonfinite_numeric_input_raises() -> None:
    numeric = np.array([[0.1, 1.0], [np.nan, 2.0], [0.3, 3.0]])
    with pytest.raises(ValueError, match="finite"):
        DLTensorEncoder(_dataset()).transform(_partition(numeric=numeric))


def test_numeric_shape_is_validated() -> None:
    bad_numeric = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Numeric input must have shape"):
        DLTensorEncoder(_dataset()).transform(_partition(numeric=bad_numeric))


def test_categorical_shape_is_validated() -> None:
    bad_categorical = np.ones((3, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="Categorical input must have shape"):
        DLTensorEncoder(_dataset()).transform(
            _partition(categorical=bad_categorical)
        )


def test_numeric_and_categorical_row_counts_must_match() -> None:
    categorical = np.ones((2, 5), dtype=np.int64)
    with pytest.raises(ValueError, match="equal row counts"):
        DLTensorEncoder(_dataset()).transform(
            _partition(categorical=categorical)
        )


def test_categorical_ids_must_be_integer() -> None:
    categorical = _partition().categorical.astype(np.float32)
    with pytest.raises(TypeError, match="integer dtype"):
        DLTensorEncoder(_dataset()).transform(
            _partition(categorical=categorical)
        )


def test_out_of_range_categorical_id_raises() -> None:
    categorical = _partition().categorical.copy()
    categorical[0, 2] = 99  # carrier ID is outside its fitted table.
    with pytest.raises(ValueError, match="carrier"):
        DLTensorEncoder(_dataset()).transform(
            _partition(categorical=categorical)
        )


def test_metadata_row_count_must_match() -> None:
    metadata = _metadata([0, 1])
    with pytest.raises(ValueError, match="metadata"):
        DLTensorEncoder(_dataset()).transform(
            _partition(metadata=metadata)
        )


def test_labels_must_be_binary() -> None:
    partition = _partition(labels=[0, 2, 1])
    with pytest.raises(ValueError, match="binary"):
        DLTensorEncoder(_dataset(partition)).transform(partition)


def test_fractional_labels_are_not_silently_truncated() -> None:
    partition = _partition()
    partition.metadata["y_true"] = partition.metadata["y_true"].astype(float)
    partition.metadata.loc[1, "y_true"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        DLTensorEncoder(_dataset(partition)).transform(partition)


def test_embedding_tables_use_dataset_sizes_and_custom_dimensions() -> None:
    sizes = _dataset().embedding_sizes
    tables = build_embedding_tables(
        sizes,
        embedding_dims={"carrier": 5, "origin": 3, "dest": 4},
    )

    assert tuple(tables) == EMBEDDING_FIELDS
    assert tables["carrier"].num_embeddings == sizes["carrier"]
    assert tables["carrier"].embedding_dim == 5
    assert tables["origin"].embedding_dim == 3
    assert tables["dest"].embedding_dim == 4


def test_default_embedding_dimensions_are_compact_and_valid() -> None:
    dimensions = default_embedding_dimensions(_dataset().embedding_sizes)
    assert set(dimensions) == set(EMBEDDING_FIELDS)
    assert all(2 <= width <= 16 for width in dimensions.values())


def test_unseen_embedding_row_is_zero_and_frozen() -> None:
    tables = build_embedding_tables(_dataset().embedding_sizes)
    for field in EMBEDDING_FIELDS:
        table = tables[field]
        assert table.padding_idx == 1
        assert torch.count_nonzero(table.weight[1]).item() == 0

        table(torch.tensor([1, 2], dtype=torch.long)).sum().backward()
        assert torch.count_nonzero(table.weight.grad[1]).item() == 0


def test_missing_embedding_row_remains_trainable() -> None:
    tables = build_embedding_tables(_dataset().embedding_sizes)
    table = tables["carrier"]
    table(torch.tensor([0], dtype=torch.long)).sum().backward()
    assert torch.count_nonzero(table.weight.grad[0]).item() > 0


def test_embedding_builder_rejects_missing_or_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="Missing embedding sizes"):
        build_embedding_tables({"carrier": 4, "origin": 4})
    with pytest.raises(ValueError, match="at least 2"):
        build_embedding_tables({"carrier": 1, "origin": 4, "dest": 4})


def test_embedding_builder_rejects_invalid_dimensions() -> None:
    sizes = _dataset().embedding_sizes
    with pytest.raises(ValueError, match="positive"):
        build_embedding_tables(sizes, embedding_dims={"carrier": 0})
    with pytest.raises(ValueError, match="Unexpected"):
        build_embedding_tables(sizes, embedding_dims={"route": 4})


def test_encoder_embeddings_integrate_with_samikshas_mlp_and_backpropagate() -> None:
    dataset = _dataset()
    batch = DLTensorEncoder(dataset).transform_partition("train")
    tables = build_embedding_tables(
        dataset.embedding_sizes,
        embedding_dims={"carrier": 4, "origin": 3, "dest": 2},
    )
    model = FlightDelayMLP(
        dense_dim=batch.x_dense.shape[1],
        carrier_embedding=tables["carrier"],
        origin_embedding=tables["origin"],
        dest_embedding=tables["dest"],
    )

    logits = model(batch.x_dense, batch.x_cat)
    assert tuple(logits.shape) == (3,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.labels
    )
    loss.backward()

    for field in EMBEDDING_FIELDS:
        assert tables[field].weight.grad is not None
