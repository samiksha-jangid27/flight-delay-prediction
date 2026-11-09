import pytest

torch = pytest.importorskip("torch")
from torch import nn

from flight_delay.deep_learning.mlp import EMBEDDING_FIELDS, FlightDelayMLP


def make_model(dense_dim=4, **kwargs):
    return FlightDelayMLP(
        dense_dim,
        carrier_embedding=nn.Embedding(5, 4, padding_idx=0),
        origin_embedding=nn.Embedding(4, 2, padding_idx=0),
        dest_embedding=nn.Embedding(6, 8, padding_idx=0),
        **kwargs,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_forward_returns_one_finite_logit_per_flight(batch_size):
    model = make_model(hidden_dims=(8,), dropout=0.2)
    x_dense = torch.randn(batch_size, 4)
    # The upstream dataset reserves 0 for missing and 1 for unseen values.
    x_cat = torch.tensor([[0, 1, 5]]).repeat(batch_size, 1)

    logits = model(x_dense, x_cat)

    assert logits.shape == (batch_size,)
    assert torch.isfinite(logits).all()
    assert model.network[0].in_features == 4 + 4 + 2 + 8
    assert model.network[0].out_features == 8
    assert model.network[2].p == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("dense_shape", "cat_shape", "message"),
    [
        ((4,), (1, 3), "x_dense must have shape"),
        ((2, 5), (2, 3), "x_dense must have shape"),
        ((2, 4), (3,), "x_cat must have shape"),
        ((2, 4), (2, 5), "x_cat must have shape"),
        ((2, 4), (3, 3), "same batch size"),
    ],
)
def test_forward_rejects_invalid_shapes(dense_shape, cat_shape, message):
    with pytest.raises(ValueError, match=message):
        make_model()(torch.zeros(dense_shape), torch.zeros(cat_shape, dtype=torch.long))


def test_forward_rejects_invalid_dtypes():
    model = make_model()
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
    with pytest.raises(TypeError, match="torch.long"):
        model(torch.zeros(2, 4), torch.zeros(2, 3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dense_dim": 0}, "dense_dim"),
        ({"hidden_dims": ()}, "hidden_dims"),
        ({"hidden_dims": (8, 0)}, "hidden_dims"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_model(**kwargs)


def test_binary_loss_backpropagates_into_all_three_embeddings():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        model = make_model(dropout=0.0)
        x_dense = torch.randn(4, 4)
        x_cat = torch.tensor([[2, 2, 3], [3, 3, 2], [4, 2, 4], [2, 3, 5]])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = nn.BCEWithLogitsLoss()(model(x_dense, x_cat), labels)
        loss.backward()

    for field in EMBEDDING_FIELDS:
        grad = model.embeddings[field].weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0


def test_mlp_learns_an_interaction_between_carrier_and_dense_input():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        model = make_model(dense_dim=1, hidden_dims=(16, 8), dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        loss_fn = nn.BCEWithLogitsLoss()
        # The label is XOR of the dense bit and the carrier bit.
        # Both inputs are needed, and origin/dest vary independently.
        category_ids = torch.arange(2, 4)
        rows = torch.cartesian_prod(torch.arange(2), category_ids, category_ids, category_ids)
        x_dense = rows[:, :1].float()
        x_cat = rows[:, 1:]
        labels = (rows[:, 0] ^ (x_cat[:, 0] - 2)).float()

        for _ in range(100):
            optimizer.zero_grad()
            loss = loss_fn(model(x_dense, x_cat), labels)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_dense, x_cat)
        assert loss_fn(logits, labels).item() < 0.2
        assert torch.equal(logits > 0, labels.bool())
