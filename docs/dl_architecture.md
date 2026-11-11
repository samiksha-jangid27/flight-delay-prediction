# Flight-delay MLP architecture

`FlightDelayMLP` predicts whether a flight will arrive at least 15 minutes
late. It learns a separate embedding for each airline, origin airport and
destination airport. The dense layers can combine those vectors with schedule
and historical inputs. For example, the model can learn that the association
between scheduled departure time and delay risk varies by destination.

## Batch interface

| Value | Shape | Meaning |
| --- | --- | --- |
| `x_dense` | `(batch, dense_dim)` | Float32 numeric features and any other dense encodings, in the encoder's fixed column order |
| `x_cat` | `(batch, 3)` | Torch long IDs for `carrier`, `origin`, `dest`, in that order |
| Return value | `(batch,)` | One raw logit per flight, including for a batch of one |

Use the train-fitted encodings from the [DL data contract](dl_data_contract.md).
The dataset adapter returns 26 numeric features and five categorical columns.
`DLTensorEncoder` puts the three embedded fields in `x_cat` and appends
one-hot blocks for `route` and `distance_band` to `x_dense`. ID 0 activates
the missing-value column. ID 1 produces an all-zero block so an unseen
value cannot activate a column with untrained weights. The encoder supplies
`dense_dim` and returns labels and metadata aligned with both input tensors.

`build_embedding_tables` creates the three tables from the encoder's
train-fitted sizes. By default, the embedding for unseen ID 1 is a frozen
zero vector. The missing ID 0 remains trainable. The MLP registers all
three tables as part of its parameters and checkpoint.

## Run a batch through the MLP

Run this example from the repository root after installing the optional `dl`
dependencies. It passes 64 training flights through a newly initialized model.

```python
import torch

from flight_delay.deep_learning.dataset import build_dl_dataset
from flight_delay.deep_learning.embeddings import build_embedding_tables
from flight_delay.deep_learning.encoding import DLTensorEncoder
from flight_delay.deep_learning.mlp import FlightDelayMLP

dataset = build_dl_dataset("data/raw/flights.csv")
encoder = DLTensorEncoder(dataset)
tables = build_embedding_tables(encoder.embedding_sizes)
model = FlightDelayMLP(
    encoder.dense_dim,
    carrier_embedding=tables["carrier"],
    origin_embedding=tables["origin"],
    dest_embedding=tables["dest"],
)

batch = encoder.transform_partition("train")
model.eval()
with torch.inference_mode():
    logits = model(batch.x_dense[:64], batch.x_cat[:64])
print(logits.shape)  # torch.Size([64])
```

## Network and training handoff

The model concatenates `x_dense` and the three embedding vectors. It applies
each configured hidden linear layer followed by ReLU and dropout, then a
linear layer with one output. The starting hidden widths are 64 and 32 with
dropout probability 0.1. These are initial settings for validation experiments.
The embedding widths come from the supplied tables.

Train with `BCEWithLogitsLoss` and float labels shaped `(batch,)`. Use
`model.eval()` and apply sigmoid to the logits when exporting probabilities.
Save the fitted vocabulary, numeric preprocessing statistics, feature order,
and model settings alongside the model state so a checkpoint can be
reconstructed with the same input order and table sizes.

## Verification

```bash
python -m pip install -e ".[dl]"
python -m pytest -q tests/test_dl_dataset.py tests/test_dl_encoding.py tests/test_dl_mlp.py
```

The MLP tests require the optional `dl` extra and skip without PyTorch.
The tests cover single-flight and multi-flight batches, missing and unseen IDs,
invalid inputs, gradients into each embedding table, and learning a synthetic
interaction that depends on both the carrier and a dense input. A one-hot MLP
with the same source features and training protocol can isolate the effect of
the learned embeddings on the real validation set.
