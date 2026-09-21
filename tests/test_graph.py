import numpy as np
import pytest
import torch

from tsgf.data import N_STATIC_NODES, N_VARS, Preprocessor
from tsgf.graph import GraphSpec, association_matrices, build_edge_index, describe

N = N_VARS + N_STATIC_NODES


@pytest.fixture(scope="module")
def assoc(synthetic_raw):
    d = Preprocessor().fit(synthetic_raw, np.arange(200)).transform(synthetic_raw, np.arange(200))
    return association_matrices(d["var_mean"], d["static_raw"], d["static_cat_raw"], min_pairs=20)


def _deg(ei):
    return torch.bincount(ei[0], minlength=N)


def test_association_shapes_and_range(assoc):
    A_vv, A_sv = assoc
    assert A_vv.shape == (N_VARS, N_VARS) and A_sv.shape == (N_STATIC_NODES, N_VARS)
    assert np.allclose(A_vv, A_vv.T) and np.all(np.diag(A_vv) == 0)
    assert A_vv.min() >= 0 and A_vv.max() <= 1 and A_sv.min() >= 0 and A_sv.max() <= 1


def test_informed_graph_is_symmetric_without_self_loops(assoc):
    ei = build_edge_index(*assoc, GraphSpec("corr_knn", 4, 6))
    assert ei.shape[0] == 2 and ei.max() < N
    assert not (ei[0] == ei[1]).any()
    fwd = {tuple(e) for e in ei.t().tolist()}
    assert all((j, i) in fwd for i, j in fwd)
    assert len(fwd) == ei.shape[1]                             # no duplicates
    assert describe(ei)["n_edges_cross_modal"] > 0
    assert (_deg(ei)[:N_VARS] >= 4).all()                      # every variable keeps >= k_var neighbours


def test_static_node_reaches_exactly_k_variables(assoc):
    ei = build_edge_index(*assoc, GraphSpec("cross_only", 4, 6))
    assert (_deg(ei)[N_VARS:] == 6).all()
    assert ((ei[0] >= N_VARS) ^ (ei[1] >= N_VARS)).all()       # every edge crosses modalities
    ei = build_edge_index(*assoc, GraphSpec("varvar_only", 4, 6))
    assert (ei < N_VARS).all()


def test_none_and_full(assoc):
    assert build_edge_index(*assoc, GraphSpec("none")).shape == (2, 0)
    assert build_edge_index(*assoc, GraphSpec("full")).shape == (2, N * (N - 1))


def test_random_graph_matches_edge_counts_but_not_edges(assoc):
    inf = build_edge_index(*assoc, GraphSpec("corr_knn", 4, 6))
    rnd = build_edge_index(*assoc, GraphSpec("random", 4, 6), seed=1)
    assert rnd.shape == inf.shape
    assert describe(rnd)["n_edges_cross_modal"] == describe(inf)["n_edges_cross_modal"]
    assert {tuple(e) for e in rnd.t().tolist()} != {tuple(e) for e in inf.t().tolist()}
    assert torch.equal(rnd, build_edge_index(*assoc, GraphSpec("random", 4, 6), seed=1))   # seeded


def test_permuted_graph_preserves_degree_sequence(assoc):
    inf = build_edge_index(*assoc, GraphSpec("corr_knn", 4, 6))
    per = build_edge_index(*assoc, GraphSpec("permuted", 4, 6), seed=2)
    assert per.shape == inf.shape
    assert torch.equal(_deg(per)[:N_VARS].sort().values, _deg(inf)[:N_VARS].sort().values)
    assert torch.equal(_deg(per)[N_VARS:].sort().values, _deg(inf)[N_VARS:].sort().values)
    assert not torch.equal(per, inf)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        GraphSpec("bogus")
