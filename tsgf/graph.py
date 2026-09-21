"""Graph construction.

Node set (fixed, shared by all patients):
    nodes 0 .. V-1          one node per dynamic variable ("temporal nodes")
    nodes V .. V+S-1        one node per static descriptor ("static nodes")

Edges are a dataset-level *association prior* estimated from the TRAINING split only:
    variable--variable   the k strongest |Spearman| associations between per-patient
                         variable means (undirected, symmetrised)
    static--variable     for every static node, its k most associated variables
                         (|Spearman| between the descriptor and the variable mean;
                         categorical descriptors use the max over their indicator columns)

The same module builds the control graphs used to ask whether the structure matters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .data import DYNAMIC_VARS, N_STATIC_NODES, N_VARS, STATIC_NODES

KINDS = ("corr_knn", "varvar_only", "cross_only", "none", "full", "random", "permuted")


@dataclass
class GraphSpec:
    kind: str = "corr_knn"
    k_var: int = 4
    k_static: int = 6
    min_pairs: int = 30          # min. patients with both quantities observed
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown graph kind {self.kind!r}; expected one of {KINDS}")


def association_matrices(var_mean: np.ndarray, static_raw: np.ndarray,
                        static_cat_raw: np.ndarray, min_pairs: int = 30):
    """|Spearman| associations from training patients.

    var_mean       (n, V)  per-patient mean of each z-scored variable, NaN = never measured
    static_raw     (n, 3)  Age, Height, Weight (NaN = missing)
    static_cat_raw (n, 2)  Gender (0/1), ICUType (1-4), NaN = missing
    Returns A_vv (V, V) with zero diagonal and A_sv (S, V).
    """
    cols = {f"v{i}": var_mean[:, i] for i in range(N_VARS)}
    groups: dict[int, list[str]] = {}
    for s, (name, kind) in enumerate(STATIC_NODES):
        if kind == "num":
            col = {"Age": 0, "Height": 1, "Weight": 2}[name]
            cols[f"s{s}"] = static_raw[:, col]
            groups[s] = [f"s{s}"]
        elif name == "Gender":
            g = static_cat_raw[:, 0]
            cols[f"s{s}"] = np.where(np.isnan(g), np.nan, (g == 1).astype(float))
            groups[s] = [f"s{s}"]
        else:  # ICUType: one indicator column per level
            u = static_cat_raw[:, 1]
            names = []
            for lvl in (1, 2, 3, 4):
                cols[f"s{s}_{lvl}"] = np.where(np.isnan(u), np.nan, (u == lvl).astype(float))
                names.append(f"s{s}_{lvl}")
            groups[s] = names
    C = pd.DataFrame(cols).corr(method="spearman", min_periods=min_pairs).abs().fillna(0.0)
    vcols = [f"v{i}" for i in range(N_VARS)]
    A_vv = C.loc[vcols, vcols].to_numpy().copy()
    np.fill_diagonal(A_vv, 0.0)
    A_sv = np.stack([C.loc[groups[s], vcols].to_numpy().max(0) for s in range(N_STATIC_NODES)])
    return A_vv, A_sv


def _knn_vv(A: np.ndarray, k: int) -> set[tuple[int, int]]:
    pairs = set()
    for i in range(len(A)):
        for j in np.argsort(-A[i], kind="stable")[:k]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return pairs


def _knn_sv(A: np.ndarray, k: int) -> set[tuple[int, int]]:
    return {(s, int(c)) for s in range(len(A)) for c in np.argsort(-A[s], kind="stable")[:k]}


def _to_edge_index(vv: set, sv: set) -> torch.Tensor:
    src, dst = [], []
    for i, j in vv:
        src += [i, j]; dst += [j, i]
    for s, c in sv:
        src += [N_VARS + s, c]; dst += [c, N_VARS + s]
    if not src:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def build_edge_index(A_vv: np.ndarray, A_sv: np.ndarray, spec: GraphSpec,
                     seed: int = 0) -> torch.Tensor:
    """(2, E) directed edge list (both directions of every undirected edge), no self loops."""
    n = N_VARS + N_STATIC_NODES
    rng = np.random.default_rng(seed)
    if spec.kind == "none":
        return _to_edge_index(set(), set())
    if spec.kind == "full":
        idx = np.array([(i, j) for i in range(n) for j in range(n) if i != j]).T
        return torch.from_numpy(idx).long()

    vv = _knn_vv(A_vv, spec.k_var)
    sv = _knn_sv(A_sv, spec.k_static)
    if spec.kind == "corr_knn":
        return _to_edge_index(vv, sv)
    if spec.kind == "varvar_only":
        return _to_edge_index(vv, set())
    if spec.kind == "cross_only":
        return _to_edge_index(set(), sv)
    if spec.kind == "permuted":
        # same topology and degree sequence, but node identities are shuffled within type
        pv = rng.permutation(N_VARS)
        ps = rng.permutation(N_STATIC_NODES)
        vv = {(min(int(pv[i]), int(pv[j])), max(int(pv[i]), int(pv[j]))) for i, j in vv}
        sv = {(int(ps[s]), int(pv[c])) for s, c in sv}
        return _to_edge_index(vv, sv)
    if spec.kind == "random":
        # same number of edges of each type, endpoints drawn uniformly at random
        all_vv = [(i, j) for i in range(N_VARS) for j in range(i + 1, N_VARS)]
        all_sv = [(s, c) for s in range(N_STATIC_NODES) for c in range(N_VARS)]
        pick_vv = rng.choice(len(all_vv), size=len(vv), replace=False)
        pick_sv = rng.choice(len(all_sv), size=len(sv), replace=False)
        return _to_edge_index({all_vv[i] for i in pick_vv}, {all_sv[i] for i in pick_sv})
    raise AssertionError(spec.kind)


def describe(edge_index: torch.Tensor) -> dict:
    """Small summary for logs / result tables."""
    n = N_VARS + N_STATIC_NODES
    e = edge_index.shape[1]
    cross = int(((edge_index[0] >= N_VARS) ^ (edge_index[1] >= N_VARS)).sum()) // 2
    return {"n_nodes": n, "n_edges_directed": e, "n_edges_cross_modal": cross,
            "mean_degree": round(e / n, 2)}


def top_edges(A_vv: np.ndarray, A_sv: np.ndarray, spec: GraphSpec, n: int = 10) -> list[str]:
    """Human-readable strongest edges of the informed graph (for sanity checks / docs)."""
    out = []
    for s, c in _knn_sv(A_sv, spec.k_static):
        out.append((A_sv[s, c], f"{STATIC_NODES[s][0]} -- {DYNAMIC_VARS[c]}"))
    for i, j in _knn_vv(A_vv, spec.k_var):
        out.append((A_vv[i, j], f"{DYNAMIC_VARS[i]} -- {DYNAMIC_VARS[j]}"))
    return [f"{name} ({w:.2f})" for w, name in sorted(out, reverse=True)[:n]]
