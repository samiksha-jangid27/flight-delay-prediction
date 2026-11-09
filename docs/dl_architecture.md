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
Select the three embedded columns by name using `EMBEDDING_FIELDS`. For a
comparison using all 31 source features, one-hot encode the remaining
`route` and `distance_band` columns using the training vocabularies and
append them to `x_dense`. The encoder supplies the resulting `dense_dim`.
Keep metadata and labels in the same row order as both inputs.

Pass the encoder's three `nn.Embedding` tables as `carrier_embedding`,
`origin_embedding` and `dest_embedding`. Size each table from the dataset's
`embedding_sizes` so that it has rows for ID 0 (missing) and ID 1 (unseen).
The model looks up those IDs as supplied and registers the tables as part of
its parameters and checkpoint. The encoder owns the vocabulary and how the
reserved embedding rows are initialized or masked.

## Network and training handoff

The model concatenates `x_dense` and the three embedding vectors. It applies
each configured hidden linear layer followed by ReLU and dropout, then a
linear layer with one output. The starting hidden widths are 64 and 32 with
dropout probability 0.1. These are initial settings for validation experiments.
The embedding widths come from the supplied tables.

Train with `BCEWithLogitsLoss` and float labels shaped `(batch,)`. Use
`model.eval()` and apply sigmoid to the logits when exporting probabilities.
Save the fitted encoder and model settings alongside the model state so a
checkpoint can be reconstructed with the same input order and table sizes.

## Verification

```bash
python -m pip install -e ".[dl]"
python -m pytest -q tests/test_dl_mlp.py
```

The MLP tests require the optional `dl` extra and skip without PyTorch.
The tests cover single-flight and multi-flight batches, missing and unseen IDs,
invalid inputs, gradients into each embedding table, and learning a synthetic
interaction that depends on both the carrier and a dense input. A one-hot MLP
with the same source features and training protocol can isolate the effect of
the learned embeddings on the real validation set.
