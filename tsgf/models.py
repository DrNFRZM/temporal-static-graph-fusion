"""Models. Every model exposes `embed(batch) -> (B, d_hidden)` and a linear `head`.

Batch = dict of tensors (see data.Preprocessor.transform):
    x (B, T, V, 2)   LOCF z-scored value + observed flag, per variable
    f (B, 5V)        per-variable summary features (mean/min/max/last/frac observed)
    s_vec (B, 14)    static descriptors as a flat vector
    s_num (B, 3), s_miss (B, 3), s_cat (B, 2)   static descriptors, tokenised form
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from .data import N_GENDER, N_ICU, N_NUM, N_STATIC_NODES, N_VARS

S_VEC_DIM = N_NUM + N_NUM + N_GENDER + N_ICU   # 14
F_DIM = 5 * N_VARS                              # 180


def mlp(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout))


class VariableGRUEncoder(nn.Module):
    """One GRU whose weights are shared across variables; each variable is encoded on its own.

    (B, T, V, 2) -> (B, V, d). A learned per-variable embedding is added to the final hidden
    state so the shared encoder output stays variable-specific.
    """

    def __init__(self, n_vars: int, d: int, in_ch: int = 2):
        super().__init__()
        self.gru = nn.GRU(in_ch, d, batch_first=True)
        self.var_emb = nn.Parameter(torch.randn(n_vars, d) * 0.1)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, C = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(B * V, T, C)
        _, h = self.gru(seq)
        return self.norm(h[-1].view(B, V, -1) + self.var_emb)


class StaticTokenizer(nn.Module):
    """Static descriptors -> one d-dim token each: (B, S, d), order = data.STATIC_NODES."""

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(N_NUM, d) * 0.5)
        self.b = nn.Parameter(torch.zeros(N_NUM, d))
        self.miss = nn.Parameter(torch.randn(N_NUM, d) * 0.1)
        self.gender = nn.Embedding(N_GENDER, d)
        self.icu = nn.Embedding(N_ICU, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch: dict) -> torch.Tensor:
        num = batch["s_num"].unsqueeze(-1) * self.w + self.b + batch["s_miss"].unsqueeze(-1) * self.miss
        cat = torch.stack([self.gender(batch["s_cat"][:, 0]), self.icu(batch["s_cat"][:, 1])], dim=1)
        return self.norm(torch.cat([num, cat], dim=1))


class Base(nn.Module):
    def __init__(self, d_hidden: int):
        super().__init__()
        self.head = nn.Linear(d_hidden, 1)

    def embed(self, b: dict) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, b: dict) -> torch.Tensor:
        return self.head(self.embed(b)).squeeze(-1)


class StaticMLP(Base):
    def __init__(self, d_hidden: int, dropout: float, **_):
        super().__init__(d_hidden)
        self.net = nn.Sequential(mlp(S_VEC_DIM, d_hidden, dropout), mlp(d_hidden, d_hidden, dropout))

    def embed(self, b):
        return self.net(b["s_vec"])


class ConcatMLP(Base):
    """Baseline 1: hand-crafted temporal summaries and static descriptors concatenated."""

    def __init__(self, d_hidden: int, dropout: float, **_):
        super().__init__(d_hidden)
        self.net = nn.Sequential(mlp(F_DIM + S_VEC_DIM, d_hidden, dropout), mlp(d_hidden, d_hidden, dropout))

    def embed(self, b):
        return self.net(torch.cat([b["f"], b["s_vec"]], dim=-1))


class TemporalOnly(Base):
    def __init__(self, d_node: int, d_hidden: int, dropout: float, **_):
        super().__init__(d_hidden)
        self.enc = VariableGRUEncoder(N_VARS, d_node)
        self.net = nn.Sequential(nn.Flatten(), mlp(N_VARS * d_node, d_hidden, dropout),
                                 mlp(d_hidden, d_hidden, dropout))

    def embed(self, b):
        return self.net(self.enc(b["x"]))


class LateFusion(Base):
    """Baseline 2: independent temporal encoder and static MLP, concatenated embeddings."""

    def __init__(self, d_node: int, d_hidden: int, dropout: float, **_):
        super().__init__(d_hidden)
        self.enc = VariableGRUEncoder(N_VARS, d_node)
        self.t_proj = nn.Sequential(nn.Flatten(), mlp(N_VARS * d_node, d_hidden, dropout))
        self.s_mlp = mlp(S_VEC_DIM, d_hidden, dropout)
        self.fuse = mlp(2 * d_hidden, d_hidden, dropout)

    def embed(self, b):
        return self.fuse(torch.cat([self.t_proj(self.enc(b["x"])), self.s_mlp(b["s_vec"])], dim=-1))


class GraphFusion(Base):
    """Temporal-static graph fusion.

    Nodes: V variable nodes (per-variable GRU embedding) + S static nodes (tokenised descriptor).
    Message passing: pre-norm residual GATv2 layers over a fixed, dataset-level edge_index.
    Readout: 'flatten' (concatenate all node states) or 'mean' (mean per node type, concatenated).
    """

    def __init__(self, edge_index: torch.Tensor, d_node: int, d_hidden: int, dropout: float,
                 gnn_layers: int = 2, gnn_heads: int = 4, readout: str = "flatten", **_):
        super().__init__(d_hidden)
        assert readout in ("flatten", "mean") and d_node % gnn_heads == 0
        self.register_buffer("edge_index", edge_index.long())
        self.n_nodes = N_VARS + N_STATIC_NODES
        self.readout_kind = readout
        self.temporal = VariableGRUEncoder(N_VARS, d_node)
        self.static = StaticTokenizer(d_node)
        self.norms = nn.ModuleList([nn.LayerNorm(d_node) for _ in range(gnn_layers)])
        self.convs = nn.ModuleList([
            # Explicit self-loops keep the behaviour stable across PyG versions. In the
            # graph_none control, nodes remain isolated but still receive this per-node update.
            GATv2Conv(d_node, d_node // gnn_heads, heads=gnn_heads, dropout=dropout,
                      add_self_loops=True)
            for _ in range(gnn_layers)])
        self.drop = nn.Dropout(dropout)
        d_read = self.n_nodes * d_node if readout == "flatten" else 2 * d_node
        self.readout = nn.Sequential(mlp(d_read, d_hidden, dropout), mlp(d_hidden, d_hidden, dropout))

    def batched_edge_index(self, B: int) -> torch.Tensor:
        """Replicate the shared graph B times with node-index offsets (PyG mini-batch layout)."""
        off = torch.arange(B, device=self.edge_index.device) * self.n_nodes
        return (self.edge_index.unsqueeze(1) + off.view(1, B, 1)).reshape(2, -1)

    def node_states(self, b: dict) -> torch.Tensor:
        h = torch.cat([self.temporal(b["x"]), self.static(b)], dim=1)     # (B, N, d)
        B, N, d = h.shape
        ei = self.batched_edge_index(B)
        h = h.reshape(B * N, d)
        for norm, conv in zip(self.norms, self.convs):
            h = h + self.drop(torch.nn.functional.gelu(conv(norm(h), ei)))
        return h.view(B, N, d)

    def embed(self, b):
        h = self.node_states(b)
        if self.readout_kind == "flatten":
            r = h.flatten(1)
        else:
            r = torch.cat([h[:, :N_VARS].mean(1), h[:, N_VARS:].mean(1)], dim=-1)
        return self.readout(r)


def build_model(kind: str, mcfg: dict, edge_index: torch.Tensor | None = None) -> Base:
    kw = dict(d_node=mcfg["d_node"], d_hidden=mcfg["d_hidden"], dropout=mcfg["dropout"])
    if kind == "static_mlp":
        return StaticMLP(**kw)
    if kind == "concat_mlp":
        return ConcatMLP(**kw)
    if kind == "temporal_only":
        return TemporalOnly(**kw)
    if kind == "late_fusion":
        return LateFusion(**kw)
    if kind == "graph":
        assert edge_index is not None
        return GraphFusion(edge_index, gnn_layers=mcfg.get("gnn_layers", 2),
                           gnn_heads=mcfg.get("gnn_heads", 4),
                           readout=mcfg.get("readout", "flatten"), **kw)
    raise ValueError(f"unknown model kind {kind!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
