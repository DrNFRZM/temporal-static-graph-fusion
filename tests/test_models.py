import numpy as np
import pytest
import torch

from tsgf.data import N_STATIC_NODES, N_VARS, Preprocessor
from tsgf.graph import GraphSpec, association_matrices, build_edge_index
from tsgf.models import GraphFusion, build_model, n_params
from tsgf.train import take, to_device

MCFG = {"d_node": 16, "d_hidden": 32, "dropout": 0.1, "gnn_layers": 2, "gnn_heads": 4}
N = N_VARS + N_STATIC_NODES


@pytest.fixture(scope="module")
def batch(synthetic_raw):
    d = Preprocessor().fit(synthetic_raw, np.arange(200)).transform(synthetic_raw, np.arange(200))
    A = association_matrices(d["var_mean"], d["static_raw"], d["static_cat_raw"], min_pairs=20)
    return take(to_device(d, torch.device("cpu")), torch.arange(8)), A


@pytest.mark.parametrize("kind", ["static_mlp", "concat_mlp", "temporal_only", "late_fusion", "graph"])
def test_forward_shapes_and_gradients(batch, kind):
    b, A = batch
    m = build_model(kind, MCFG, build_edge_index(*A, GraphSpec("corr_knn", 4, 6)))
    assert m.embed(b).shape == (8, 32)
    logits = m(b)
    assert logits.shape == (8,)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, b["y"]).backward()
    assert all(p.grad is not None for p in m.parameters())


@pytest.mark.parametrize("spec", ["none", "full", "random", "permuted", "varvar_only", "cross_only"])
def test_graph_model_runs_on_every_graph_kind(batch, spec):
    b, A = batch
    m = build_model("graph", MCFG, build_edge_index(*A, GraphSpec(spec, 4, 6)))
    assert torch.isfinite(m(b)).all()


def test_node_state_shape_and_meanpool_readout(batch):
    b, A = batch
    m = build_model("graph", {**MCFG, "readout": "mean"}, build_edge_index(*A, GraphSpec()))
    assert m.node_states(b).shape == (8, N, 16)
    assert m.embed(b).shape == (8, 32)


def test_none_control_keeps_only_self_updates(batch):
    """With no inter-node edges, changing one node cannot alter another node's state."""
    b, A = batch
    m = build_model("graph", MCFG, build_edge_index(*A, GraphSpec("none"))).eval()
    changed = {k: v.clone() for k, v in b.items()}
    changed["x"][:, :, 0, :] += 10.0
    with torch.no_grad():
        before, after = m.node_states(b), m.node_states(changed)
    assert all(conv.add_self_loops for conv in m.convs)
    assert not torch.allclose(before[:, 0], after[:, 0])
    assert torch.allclose(before[:, 1:], after[:, 1:], atol=1e-6)


def test_batched_edge_index_offsets(batch):
    _, A = batch
    ei = build_edge_index(*A, GraphSpec())
    be = GraphFusion(ei, 16, 32, 0.1).batched_edge_index(3)
    E = ei.shape[1]
    assert be.shape == (2, 3 * E)
    assert torch.equal(be[:, E:2 * E], ei + N) and torch.equal(be[:, 2 * E:], ei + 2 * N)


def test_samples_do_not_interact(batch):
    """Batched message passing must not leak information between patients."""
    b, A = batch
    m = build_model("graph", MCFG, build_edge_index(*A, GraphSpec())).eval()
    with torch.no_grad():
        full = m(b)
        single = m(take(b, torch.tensor([3])))
    assert torch.allclose(full[3], single[0], atol=1e-5)


def test_same_seed_same_init():
    torch.manual_seed(0); a = build_model("late_fusion", MCFG)
    torch.manual_seed(0); c = build_model("late_fusion", MCFG)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))
    assert n_params(a) > 0
