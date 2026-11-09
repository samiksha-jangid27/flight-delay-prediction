"""Convert the fitted DL dataset partitions into MLP-ready tensors.

The dataset adapter owns train-only vocabulary fitting, numeric imputation,
and numeric standardization. This module deliberately reuses those fitted
arrays and statistics instead of fitting a second preprocessing pipeline.

Dense input layout:
    [26 numeric features, route one-hot columns, distance_band one-hot columns]

Categorical input layout (required by FlightDelayMLP):
    [carrier, origin, dest]

Category IDs follow the dataset contract:
    0 = missing, 1 = unseen, 2+ = known training categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from flight_delay.deep_learning.dataset import DLDataset, DLPartition

EMBEDDING_FIELDS: Final[tuple[str, ...]] = ("carrier", "origin", "dest")
DENSE_CATEGORICAL_FIELDS: Final[tuple[str, ...]] = ("route", "distance_band")
REQUIRED_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "flight_id",
    "flight_date",
    "y_true",
    "partition",
)


@dataclass(frozen=True)
class TensorBatch:
    """Position-aligned MLP inputs, labels, and comparison metadata.

    Tensors are returned on CPU. The training loop is responsible for moving
    them to CPU, CUDA, or Apple MPS as appropriate.
    """

    x_dense: Tensor
    x_cat: Tensor
    labels: Tensor
    metadata: pd.DataFrame

    @property
    def size(self) -> int:
        """Number of flights in this batch."""
        return int(self.labels.shape[0])


class DLTensorEncoder:
    """Adapt an already-fitted :class:`DLDataset` to the FlightDelayMLP API.

    No ``fit`` method is provided intentionally: ``build_dl_dataset`` has
    already fitted vocabularies and numeric statistics on the early training
    partition only. Every partition must be transformed using that same
    DLDataset so validation/test data cannot influence preprocessing.
    """

    def __init__(self, dataset: DLDataset) -> None:
        if not isinstance(dataset, DLDataset):
            raise TypeError("dataset must be a DLDataset")

        self.dataset = dataset
        self.numeric_features = tuple(dataset.numeric_features)
        self.categorical_features = tuple(dataset.categorical_features)

        if not self.numeric_features:
            raise ValueError("Dataset must contain at least one numeric feature")
        if len(set(self.numeric_features)) != len(self.numeric_features):
            raise ValueError("Numeric feature names must be unique")
        if len(set(self.categorical_features)) != len(self.categorical_features):
            raise ValueError("Categorical feature names must be unique")

        required_fields = EMBEDDING_FIELDS + DENSE_CATEGORICAL_FIELDS
        missing_fields = sorted(set(required_fields) - set(self.categorical_features))
        if missing_fields:
            raise ValueError(
                f"Dataset is missing required categorical features: {missing_fields}"
            )

        self.categorical_positions = {
            field: self.categorical_features.index(field)
            for field in required_fields
        }

        # Copy fitted vocabulary mappings so accidental caller mutation does
        # not change the encoder's interpretation after construction.
        self.vocabularies: dict[str, dict[str, int]] = {}
        self.embedding_sizes: dict[str, int] = {}
        dataset_sizes = dataset.embedding_sizes

        for field in required_fields:
            if field not in dataset.vocabularies or field not in dataset_sizes:
                raise ValueError(f"Missing fitted vocabulary for {field!r}")

            vocabulary = dict(dataset.vocabularies[field])
            expected_ids = set(range(2, len(vocabulary) + 2))
            actual_ids = list(vocabulary.values())
            if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
                raise ValueError(
                    f"Vocabulary for {field!r} must use unique contiguous IDs starting at 2"
                )

            size = dataset_sizes[field]
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
                raise TypeError(f"Vocabulary size for {field!r} must be an integer")
            if int(size) != len(vocabulary) + 2:
                raise ValueError(
                    f"Vocabulary size for {field!r} must equal len(vocabulary) + 2"
                )

            self.vocabularies[field] = vocabulary
            self.embedding_sizes[field] = int(size)

        self.dense_feature_names = self._make_dense_feature_names()
        self.dense_dim = len(self.dense_feature_names)

    def _make_dense_feature_names(self) -> tuple[str, ...]:
        names = list(self.numeric_features)
        for field in DENSE_CATEGORICAL_FIELDS:
            names.extend((f"{field}__missing", f"{field}__unseen"))
            by_id = sorted(
                self.vocabularies[field].items(), key=lambda item: item[1]
            )
            names.extend(f"{field}__{value}" for value, _ in by_id)
        return tuple(names)

    def transform(self, partition: DLPartition) -> TensorBatch:
        """Convert one dataset partition without refitting any transformation."""
        if not isinstance(partition, DLPartition):
            raise TypeError("partition must be a DLPartition")
        if not isinstance(partition.metadata, pd.DataFrame):
            raise TypeError("Partition metadata must be a pandas DataFrame")

        missing_metadata = [
            column
            for column in REQUIRED_METADATA_COLUMNS
            if column not in partition.metadata.columns
        ]
        if missing_metadata:
            raise ValueError(f"Partition metadata is missing columns: {missing_metadata}")

        numeric = np.asarray(partition.numeric)
        categorical = np.asarray(partition.categorical)

        if numeric.ndim != 2 or numeric.shape[1] != len(self.numeric_features):
            raise ValueError(
                f"Numeric input must have shape (N, {len(self.numeric_features)})"
            )
        if categorical.ndim != 2 or categorical.shape[1] != len(self.categorical_features):
            raise ValueError(
                "Categorical input must have shape "
                f"(N, {len(self.categorical_features)})"
            )

        row_count = numeric.shape[0]
        if row_count == 0:
            raise ValueError("Partition must contain at least one row")
        if categorical.shape[0] != row_count:
            raise ValueError("Numeric and categorical arrays must have equal row counts")
        if len(partition.metadata) != row_count:
            raise ValueError("Feature arrays and metadata must have equal row counts")

        try:
            numeric = numeric.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("Numeric features must be convertible to float32") from exc
        if not np.isfinite(numeric).all():
            raise ValueError("Numeric features must contain only finite values")

        if not np.issubdtype(categorical.dtype, np.integer):
            raise TypeError("Categorical IDs must have an integer dtype")
        categorical = categorical.astype(np.int64, copy=False)

        # Validate all five categorical columns, including the two one-hot
        # columns, against their corresponding train-fitted table sizes.
        for field in self.categorical_features:
            position = self.categorical_features.index(field)
            size = self.embedding_sizes.get(field)
            if size is None:
                # The canonical feature contract has exactly five categorical
                # fields. Fail clearly if an unexpected field is introduced.
                raise ValueError(f"No fitted vocabulary size available for {field!r}")
            ids = categorical[:, position]
            if np.any(ids < 0) or np.any(ids >= size):
                raise ValueError(
                    f"Categorical IDs for {field!r} must be in [0, {size - 1}]"
                )

        # The MLP consumes exactly these three fields, in this fixed order.
        x_cat = np.column_stack(
            [
                categorical[:, self.categorical_positions[field]]
                for field in EMBEDDING_FIELDS
            ]
        ).astype(np.int64, copy=False)

        dense_parts = [numeric]
        row_indices = np.arange(row_count)
        for field in DENSE_CATEGORICAL_FIELDS:
            ids = categorical[:, self.categorical_positions[field]]
            one_hot = np.zeros(
                (row_count, self.embedding_sizes[field]), dtype=np.float32
            )
            one_hot[row_indices, ids] = 1.0
            dense_parts.append(one_hot)

        x_dense_array = np.ascontiguousarray(
            np.concatenate(dense_parts, axis=1), dtype=np.float32
        )
        if x_dense_array.shape[1] != self.dense_dim:
            raise ValueError("Constructed dense tensor does not match dense_dim")

        raw_labels = partition.metadata["y_true"].to_numpy(copy=True)
        if raw_labels.shape != (row_count,):
            raise ValueError("Labels must have shape (N,)")
        if pd.isna(raw_labels).any() or not np.isin(raw_labels, (0, 1)).all():
            raise ValueError("Labels must contain only binary values 0 and 1")
        labels = raw_labels.astype(np.int64, copy=False)

        return TensorBatch(
            x_dense=torch.from_numpy(x_dense_array),
            x_cat=torch.from_numpy(np.ascontiguousarray(x_cat)),
            labels=torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32)),
            metadata=partition.metadata.copy(deep=True),
        )

    def transform_partition(self, name: str) -> TensorBatch:
        """Transform a named partition from the fitted dataset."""
        if name not in self.dataset.partitions:
            available = list(self.dataset.partitions)
            raise ValueError(f"Unknown partition {name!r}; available: {available}")
        return self.transform(self.dataset.partitions[name])
