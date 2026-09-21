import numpy as np
import torch

from tsgf.data import Preprocessor, probe_targets, stratified_split
from tsgf.evaluate import embedding_probes
from tsgf.graph import association_matrices
from tsgf.run import run_one
from tsgf.train import to_device
from tsgf.utils import set_seed

CPU = torch.device("cpu")
CFG = {"model": {"d_node": 8, "d_hidden": 16, "dropout": 0.1, "gnn_layers": 1, "gnn_heads": 2},
       "train": {"epochs": 3, "patience": 3, "batch_size": 64, "eval_batch_size": 128, "lr": 0.005,
                 "weight_decay": 0.01},
       "probe": {"knn_k": 5}}


def _setup(raw, seed=0):
    idx = stratified_split(raw.y, seed, 0.15, 0.15)
    prep = Preprocessor().fit(raw, idx[0])
    names = ("train", "val", "test")
    sp = {k: prep.transform(raw, i) for k, i in zip(names, idx)}
    aux = {k: probe_targets(raw, i) for k, i in zip(names, idx)}
    tr = sp["train"]
    A = association_matrices(tr["var_mean"], tr["static_raw"], tr["static_cat_raw"], min_pairs=10)
    return sp, {k: to_device(v, CPU) for k, v in sp.items()}, aux, A


def test_run_one_produces_valid_metrics(synthetic_raw):
    sp, dev, aux, A = _setup(synthetic_raw)
    for exp in ({"name": "g", "model": "graph", "graph": {"kind": "corr_knn"}},
                {"name": "l", "model": "late_fusion"}, {"name": "gbm", "model": "gbm"}):
        row = run_one(exp, CFG, 0, sp, dev, aux, A, CPU)
        assert 0 <= row["test_auroc"] <= 1 and 0 <= row["test_auprc"] <= 1 and 0 <= row["val_auroc"] <= 1


def test_training_is_deterministic_on_cpu(synthetic_raw):
    sp, dev, aux, A = _setup(synthetic_raw)
    exp = {"name": "g", "model": "graph", "graph": {"kind": "random"}}
    r1 = run_one(exp, CFG, 0, sp, dev, aux, A, CPU)
    r2 = run_one(exp, CFG, 0, sp, dev, aux, A, CPU)
    assert r1["test_auroc"] == r2["test_auroc"] and r1["probe_lr_auroc"] == r2["probe_lr_auroc"]


def test_probes_recover_planted_signal():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(600, 8))
    y = (z[:, 0] + 0.3 * rng.normal(size=600) > 0).astype(float)
    los = 5 + 3 * z[:, 0] + rng.normal(size=600)
    saps = 10 + 2 * z[:, 1] + rng.normal(size=600)
    aux = lambda s: {"los": los[s], "saps": saps[s]}
    tr, te = slice(0, 400), slice(400, 600)
    out = embedding_probes(z[tr], z[te], y[tr], y[te], aux(tr), aux(te), knn_k=15)
    assert out["probe_lr_auroc"] > 0.9 and out["probe_los_auroc"] > 0.85 and out["probe_saps_r2"] > 0.6


def test_set_seed_reproducible_randomness():
    set_seed(5); a = torch.randn(3)
    set_seed(5); b = torch.randn(3)
    assert torch.equal(a, b)
