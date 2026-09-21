"""Run all experiments of a config:  python -m tsgf.run --config configs/quick.yaml

For every seed: new stratified train/val/test split -> preprocessing and graph statistics fit on
train only -> every experiment trained from scratch -> test metrics + embedding probes appended to
<output_dir>/results.csv. Finished (experiment, seed) pairs are skipped on re-run.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from . import __version__
from .data import Preprocessor, load_raw, probe_targets, stratified_split
from .evaluate import embedding_probes
from .graph import GraphSpec, association_matrices, build_edge_index, describe, top_edges
from .models import build_model, n_params
from .train import binary_metrics, fit, predict, to_device
from .utils import load_config, resolve_device, set_seed, setup_logging

log = logging.getLogger("tsgf.run")


def _gbm(tr: dict, va: dict, te: dict, seed: int) -> dict:
    """Reference: gradient boosting on the same concatenated summary + static features."""
    feats = lambda d: np.concatenate([d["f"], d["s_vec"]], axis=1)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True,
                                         validation_fraction=0.15, random_state=seed)
    clf.fit(feats(tr), tr["y"])
    m = binary_metrics(te["y"], clf.predict_proba(feats(te))[:, 1])
    m["val_auroc"] = binary_metrics(va["y"], clf.predict_proba(feats(va))[:, 1])["auroc"]
    return {"test_" + k: v for k, v in m.items() if k != "val_auroc"} | {
        "val_auroc": m["val_auroc"], "train_seconds": time.time() - t0, "n_params": np.nan}


def run_one(exp: dict, cfg: dict, seed: int, np_splits: dict, dev_splits: dict, aux: dict,
            A: tuple, device: torch.device) -> dict:
    kind = exp["model"]
    row = {"experiment": exp["name"], "group": exp.get("group", ""), "model": kind, "seed": seed}
    if kind == "gbm":
        row.update(_gbm(np_splits["train"], np_splits["val"], np_splits["test"], seed))
        return row

    edge_index = None
    if kind == "graph":
        spec = GraphSpec(**exp.get("graph", {}))
        row["graph_kind"] = spec.kind
        edge_index = build_edge_index(*A, spec, seed=seed)
        d = describe(edge_index)
        row.update(edges_directed=d["n_edges_directed"], edges_cross=d["n_edges_cross_modal"])
    mcfg = {**cfg["model"], **exp.get("model_overrides", {})}
    set_seed(seed)
    model = build_model(kind, mcfg, edge_index).to(device)
    info = fit(model, dev_splits["train"], dev_splits["val"], cfg["train"], seed)
    ebs = cfg["train"]["eval_batch_size"]
    m = binary_metrics(np_splits["test"]["y"], predict(model, dev_splits["test"], ebs))
    row.update({"test_" + k: v for k, v in m.items()})
    row.update(val_auroc=info["best_val_auroc"], best_epoch=info["best_epoch"],
               train_seconds=info["train_seconds"], n_params=n_params(model))
    z_tr = predict(model, dev_splits["train"], ebs, embeddings=True)
    z_te = predict(model, dev_splits["test"], ebs, embeddings=True)
    row.update(embedding_probes(z_tr, z_te, np_splits["train"]["y"], np_splits["test"]["y"],
                                aux["train"], aux["test"], cfg["probe"]["knn_k"], seed))
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--seeds", help="override seeds, e.g. 0,1")
    ap.add_argument("--epochs", type=int, help="override max epochs (smoke tests)")
    ap.add_argument("--output-dir", help="override output_dir")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    out = Path(args.output_dir or cfg["output_dir"])
    setup_logging(out)
    exps = [e for e in cfg["experiments"] if not args.only or e["name"] in args.only.split(",")]
    device = resolve_device(cfg["train"]["device"])
    if device.type == "cpu" and cfg["train"].get("threads"):
        torch.set_num_threads(cfg["train"]["threads"])

    env = {"tsgf": __version__, "python": platform.python_version(), "torch": torch.__version__,
           "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "platform": platform.platform()}
    import sklearn, torch_geometric
    env.update(sklearn=sklearn.__version__, torch_geometric=torch_geometric.__version__)
    log.info("environment: %s", env)
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    (out / "config_used.yaml").write_text(yaml.safe_dump({**cfg, "experiments": exps}, sort_keys=False))

    raw = load_raw(cfg["data"]["data_dir"], tuple(cfg["data"]["sets"]))
    log.info("dataset: N=%d, mortality rate %.3f, %d dynamic vars x %d h",
             len(raw), raw.y.mean(), raw.hourly.shape[2], raw.hourly.shape[1])

    csv = out / "results.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["experiment"], int(r["seed"])) for r in rows}

    for seed in cfg["seeds"]:
        todo = [e for e in exps if (e["name"], seed) not in done]
        if not todo:
            continue
        idx = stratified_split(raw.y, seed, cfg["data"]["val_frac"], cfg["data"]["test_frac"])
        prep = Preprocessor().fit(raw, idx[0])                     # train-only statistics
        np_splits = {k: prep.transform(raw, i) for k, i in zip(("train", "val", "test"), idx)}
        aux = {k: probe_targets(raw, i) for k, i in zip(("train", "val", "test"), idx)}
        tr = np_splits["train"]
        A = association_matrices(tr["var_mean"], tr["static_raw"], tr["static_cat_raw"])  # train only
        log.info("seed %d: split %s | strongest edges: %s", seed,
                 [len(i) for i in idx], "; ".join(top_edges(*A, GraphSpec(), n=6)))
        dev_splits = {k: to_device(v, device) for k, v in np_splits.items()}
        for exp in todo:
            t0 = time.time()
            row = run_one(exp, cfg, seed, np_splits, dev_splits, aux, A, device)
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv, index=False)
            log.info("seed %d | %-22s test AUROC %.4f AUPRC %.4f | val %.4f | %.0fs", seed,
                     exp["name"], row["test_auroc"], row["test_auprc"], row["val_auroc"],
                     time.time() - t0)
    log.info("done -> %s", csv)


if __name__ == "__main__":
    main()
