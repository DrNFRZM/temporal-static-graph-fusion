"""Embedding-quality probes on frozen hidden embeddings.

Probes are fit on TRAIN embeddings and scored on TEST embeddings:
  probe_lr_auroc    logistic regression on the training label (mortality)
  probe_knn_auroc   k-NN (standardised embedding) on mortality
  probe_los_auroc   logistic regression for "ICU stay >= train-median length" - a target
                    the model was never trained on
  probe_saps_r2     ridge regression R^2 for the SAPS-I severity score (never a model input)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def embedding_probes(z_tr: np.ndarray, z_te: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray,
                     aux_tr: dict, aux_te: dict, knn_k: int = 25, seed: int = 0) -> dict[str, float]:
    sc = StandardScaler().fit(z_tr)
    a, b = sc.transform(z_tr), sc.transform(z_te)
    out = {}
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(a, y_tr)
    out["probe_lr_auroc"] = float(roc_auc_score(y_te, lr.predict_proba(b)[:, 1]))
    knn = KNeighborsClassifier(n_neighbors=knn_k).fit(a, y_tr)
    out["probe_knn_auroc"] = float(roc_auc_score(y_te, knn.predict_proba(b)[:, 1]))

    los_tr, los_te = aux_tr["los"], aux_te["los"]
    ok_tr, ok_te = ~np.isnan(los_tr), ~np.isnan(los_te)
    thr = np.median(los_tr[ok_tr])
    lr2 = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(
        a[ok_tr], (los_tr[ok_tr] >= thr).astype(int))
    out["probe_los_auroc"] = float(roc_auc_score((los_te[ok_te] >= thr).astype(int),
                                                 lr2.predict_proba(b[ok_te])[:, 1]))
    s_tr, s_te = aux_tr["saps"], aux_te["saps"]
    ok_tr, ok_te = ~np.isnan(s_tr), ~np.isnan(s_te)
    rg = Ridge(alpha=10.0).fit(a[ok_tr], s_tr[ok_tr])
    out["probe_saps_r2"] = float(r2_score(s_te[ok_te], rg.predict(b[ok_te])))
    return out
