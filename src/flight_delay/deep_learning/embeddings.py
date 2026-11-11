"""Trainable embedding tables for carrier, origin, and destination IDs."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, sqrt
from numbers import Integral
from typing import Final

import torch
from torch import nn

EMBEDDING_FIELDS: Final[tuple[str, ...]] = ("carrier", "origin", "dest")
MISSING_CATEGORY_INDEX: Final[int] = 0
UNKNOWN_CATEGORY_INDEX: Final[int] = 1


def default_embedding_dimensions(embedding_sizes: Mapping[str, int]) -> dict[str, int]:
    """Choose compact default widths from known-category cardinalities.

    The square-root heuristic is intentionally conservative for this dataset:
    widths are at least 2 and at most 16. The two reserved rows (missing and
    unseen) are excluded from the cardinality used to choose the width.
    """
    _validate_embedding_sizes(embedding_sizes)
    dimensions: dict[str, int] = {}
    for field in EMBEDDING_FIELDS:
        known_categories = int(embedding_sizes[field]) - 2
        dimensions[field] = min(16, max(2, ceil(sqrt(max(known_categories, 1)))))
    return dimensions


def _validate_embedding_sizes(embedding_sizes: Mapping[str, int]) -> None:
    if not isinstance(embedding_sizes, Mapping):
        raise TypeError("embedding_sizes must be a mapping of field names to table sizes")

    missing = [field for field in EMBEDDING_FIELDS if field not in embedding_sizes]
    if missing:
        raise ValueError(f"Missing embedding sizes for fields: {missing}")

    for field in EMBEDDING_FIELDS:
        size = embedding_sizes[field]
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise TypeError(f"Embedding size for {field!r} must be an integer")
        if int(size) < 2:
            raise ValueError(
                f"Embedding size for {field!r} must be at least 2 "
                "for missing and unseen category IDs"
            )


def build_embedding_tables(
    embedding_sizes: Mapping[str, int],
    embedding_dims: Mapping[str, int] | None = None,
    *,
    freeze_unknown: bool = True,
) -> dict[str, nn.Embedding]:
    """Create the three ``nn.Embedding`` tables expected by ``FlightDelayMLP``.

    Args:
        embedding_sizes: Use ``DLDataset.embedding_sizes``. Each size includes
            rows for missing ID 0 and unseen ID 1.
        embedding_dims: Optional per-field widths. Missing entries use the
            default square-root heuristic; unknown field names are rejected.
        freeze_unknown: When true (default), ID 1 is ``padding_idx`` and its
            vector is initialized to zero and receives no gradient. ID 0 stays
            trainable, allowing the model to learn a representation for missing
            values separately from truly unseen values.

    Returns:
        A dictionary with ``carrier``, ``origin``, and ``dest`` keys. Pass the
        values to ``FlightDelayMLP`` as its three embedding arguments.
    """
    _validate_embedding_sizes(embedding_sizes)
    if not isinstance(freeze_unknown, bool):
        raise TypeError("freeze_unknown must be a bool")

    dimensions = default_embedding_dimensions(embedding_sizes)
    if embedding_dims is not None:
        if not isinstance(embedding_dims, Mapping):
            raise TypeError("embedding_dims must be a mapping when provided")
        unexpected = sorted(set(embedding_dims) - set(EMBEDDING_FIELDS))
        if unexpected:
            raise ValueError(f"Unexpected embedding dimension fields: {unexpected}")
        for field, dimension in embedding_dims.items():
            if isinstance(dimension, bool) or not isinstance(dimension, Integral):
                raise TypeError(f"Embedding dimension for {field!r} must be an integer")
            if int(dimension) < 1:
                raise ValueError(f"Embedding dimension for {field!r} must be positive")
            dimensions[field] = int(dimension)

    tables: dict[str, nn.Embedding] = {}
    for field in EMBEDDING_FIELDS:
        table_size = int(embedding_sizes[field])
        dimension = dimensions[field]
        padding_idx = UNKNOWN_CATEGORY_INDEX if freeze_unknown else None
        table = nn.Embedding(
            num_embeddings=table_size,
            embedding_dim=dimension,
            padding_idx=padding_idx,
        )
        nn.init.normal_(table.weight, mean=0.0, std=0.05)

        if freeze_unknown:
            # Explicitly enforce the documented zero representation for ID 1.
            with torch.no_grad():
                table.weight[UNKNOWN_CATEGORY_INDEX].zero_()

        tables[field] = table

    return tables
