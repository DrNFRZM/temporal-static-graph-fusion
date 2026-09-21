"""Training loop: BCE on in-hospital mortality, early stopping on validation AUROC."""
from __future__ import annotations

import copy
import logging
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

log = logging.getLogger(__name__)

TENSOR_KEYS = ("x", "f", "s_num", "s_miss", "s_cat", "s_vec", "y")


def to_device(split: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(split[k]).to(device) for k in TENSOR_KEYS}


def take(data: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in data.items()}


@torch.no_grad()
def predict(model: nn.Module, data: dict, batch_size: int, embeddings: bool = False) -> np.ndarray:
    """Probabilities (or hidden embeddings) for every sample, in order."""
    model.eval()
    n = len(data["y"])
    out = []
    for i in range(0, n, batch_size):
        b = take(data, torch.arange(i, min(i + batch_size, n), device=data["y"].device))
        out.append(model.embed(b) if embeddings else torch.sigmoid(model(b)))
    return torch.cat(out).cpu().numpy()


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def fit(model: nn.Module, train: dict, val: dict, tcfg: dict, seed: int) -> dict:
    """Train in place; restores the best-validation weights. Returns a small history dict."""
    dev = train["y"].device
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, bs = len(train["y"]), tcfg["batch_size"]
    y_val = val["y"].cpu().numpy()
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    hist = []
    for ep in range(1, tcfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        tot = 0.0
        for i in range(0, n, bs):
            b = take(train, perm[i:i + bs])
            loss = loss_fn(model(b), b["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b["y"])
        val_auc = roc_auc_score(y_val, predict(model, val, tcfg["eval_batch_size"]))
        hist.append((ep, tot / n, val_auc))
        log.debug("ep %02d  train_loss %.4f  val_auroc %.4f", ep, tot / n, val_auc)
        if val_auc > best:
            best, best_ep, bad = val_auc, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    return {"best_val_auroc": best, "best_epoch": best_ep, "epochs_run": len(hist),
            "train_seconds": time.time() - t0}
