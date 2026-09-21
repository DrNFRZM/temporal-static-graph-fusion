"""Data loading and preprocessing for the PhysioNet/CinC Challenge 2012 dataset.

Each ICU stay is one sample with
  * a *static* modality: 5 admission descriptors (Age, Gender, Height, ICUType, Weight)
  * a *temporal* modality: 36 irregularly sampled variables over the first 48 h,
    binned to hourly means -> (48, 36) array with NaN where nothing was measured.

Everything that estimates statistics (clipping bounds, means, stds) is fit on the
training split only; see `Preprocessor`.
"""
from __future__ import annotations

import io
import logging
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0"
N_HOURS = 48

# 37 time-series variables of the challenge minus Weight, which is used as a static descriptor.
DYNAMIC_VARS = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine",
    "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "MAP",
    "MechVent", "Mg", "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH",
    "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine",
    "WBC",
]
STATIC_COLS = ["Age", "Gender", "Height", "ICUType", "Weight"]
# Static node semantics used by the graph model: (name, kind)
STATIC_NODES = [("Age", "num"), ("Height", "num"), ("Weight", "num"),
                ("Gender", "cat"), ("ICUType", "cat")]
N_GENDER = 3   # female, male, missing
N_ICU = 5      # ICU types 1-4 + unknown
N_NUM = 3
N_STATIC_NODES = len(STATIC_NODES)
N_VARS = len(DYNAMIC_VARS)
_VAR_IDX = {v: i for i, v in enumerate(DYNAMIC_VARS)}
_STATIC_IDX = {c: i for i, c in enumerate(STATIC_COLS)}
OUTCOME_COLS = ["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "In-hospital_death"]


# --------------------------------------------------------------------------- download
def download(data_dir: str | Path, sets: tuple[str, ...] = ("a",)) -> None:
    """Fetch set-x.tar.gz and Outcomes-x.txt from PhysioNet (open access, no login)."""
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for s in sets:
        for fname in (f"set-{s}.tar.gz", f"Outcomes-{s}.txt"):
            dest = raw / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = f"{BASE_URL}/{fname}"
            log.info("downloading %s", url)
            urllib.request.urlretrieve(url, dest)


# --------------------------------------------------------------------------- parsing
def parse_record(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one patient file -> (static[5], hourly[48, 36]); missing = NaN.

    Static values at 00:00 use -1 as the missing marker; implausible Height/Weight are
    dropped. Dynamic values that are negative are treated as invalid (the raw files use
    -1 as an error code for several variables). Multiple measurements in one hour are averaged.
    """
    static = np.full(len(STATIC_COLS), np.nan, dtype=np.float32)
    sums = np.zeros((N_HOURS, N_VARS), dtype=np.float64)
    cnts = np.zeros((N_HOURS, N_VARS), dtype=np.int32)
    for line in text.splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        t, name, val = parts
        try:
            v = float(val)
        except ValueError:
            continue
        if name in _STATIC_IDX:
            if t == "00:00" and v >= 0:
                static[_STATIC_IDX[name]] = v
        elif name in _VAR_IDX:
            if v < 0:
                continue
            h = min(int(t.split(":")[0]), N_HOURS - 1)
            sums[h, _VAR_IDX[name]] += v
            cnts[h, _VAR_IDX[name]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        hourly = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan).astype(np.float32)
    # plausibility filter for anthropometrics
    if not 100 <= static[_STATIC_IDX["Height"]] <= 250:
        static[_STATIC_IDX["Height"]] = np.nan
    if not 20 <= static[_STATIC_IDX["Weight"]] <= 300:
        static[_STATIC_IDX["Weight"]] = np.nan
    return static, hourly


@dataclass
class RawData:
    """Unnormalised samples. Row i of every field describes the same ICU stay."""
    hourly: np.ndarray      # (N, T, V) float32, NaN = not measured
    static: pd.DataFrame    # (N, 5) columns STATIC_COLS, NaN = missing
    outcomes: pd.DataFrame  # SAPS-I, SOFA, Length_of_stay, In-hospital_death (labels / probe targets only)

    def __len__(self) -> int:
        return len(self.hourly)

    @property
    def y(self) -> np.ndarray:
        return self.outcomes["In-hospital_death"].to_numpy().astype(np.float32)


def build_raw(data_dir: str | Path, sets: tuple[str, ...]) -> RawData:
    raw = Path(data_dir) / "raw"
    hourly, static, ids, outcomes = [], [], [], []
    for s in sets:
        oc = pd.read_csv(raw / f"Outcomes-{s}.txt").set_index("RecordID")
        with tarfile.open(raw / f"set-{s}.tar.gz") as tar:
            for m in tar.getmembers():
                if not (m.isfile() and m.name.endswith(".txt")):
                    continue
                rid = int(Path(m.name).stem)
                if rid not in oc.index:
                    continue
                st, hr = parse_record(io.TextIOWrapper(tar.extractfile(m)).read())
                static.append(st); hourly.append(hr); ids.append(rid)
                outcomes.append(oc.loc[rid])
        log.info("parsed set-%s: %d records so far", s, len(ids))
    oc_df = pd.DataFrame(outcomes).reset_index(drop=True)
    oc_df.insert(0, "RecordID", ids)
    order = np.argsort(ids)
    return RawData(
        hourly=np.stack(hourly)[order],
        static=pd.DataFrame(np.stack(static)[order], columns=STATIC_COLS),
        outcomes=oc_df.iloc[order].reset_index(drop=True)[OUTCOME_COLS],
    )


def load_raw(data_dir: str | Path, sets: tuple[str, ...]) -> RawData:
    """Download (if needed), parse and cache the requested sets."""
    data_dir = Path(data_dir)
    cache = data_dir / "processed" / f"physionet2012_{''.join(sets)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return RawData(
            hourly=z["hourly"],
            static=pd.DataFrame(z["static"], columns=STATIC_COLS),
            outcomes=pd.DataFrame(z["outcomes"], columns=OUTCOME_COLS),
        )
    download(data_dir, sets)
    rd = build_raw(data_dir, sets)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, hourly=rd.hourly, static=rd.static.to_numpy(np.float32),
                        outcomes=rd.outcomes.to_numpy(np.float64))
    return rd


# --------------------------------------------------------------------------- splits
def stratified_split(y: np.ndarray, seed: int, val_frac: float, test_frac: float):
    idx = np.arange(len(y))
    tr_val, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    tr, va = train_test_split(tr_val, test_size=val_frac / (1 - test_frac),
                              stratify=y[tr_val], random_state=seed)
    return tr, va, te


# --------------------------------------------------------------------------- preprocessing
class Preprocessor:
    """Winsorise + z-score dynamic variables and static numerics; statistics from train only."""

    def fit(self, raw: RawData, idx: np.ndarray) -> "Preprocessor":
        obs = raw.hourly[idx].reshape(-1, N_VARS)
        with np.errstate(all="ignore"):
            lo, hi = np.nanquantile(obs, [0.005, 0.995], axis=0)
        lo, hi = np.nan_to_num(lo), np.nan_to_num(hi)
        clipped = np.clip(obs, lo, hi)
        with np.errstate(all="ignore"):
            mu, sd = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
        self.lo, self.hi = lo, hi
        self.mu = np.nan_to_num(mu)
        self.sd = np.where(np.nan_to_num(sd) < 1e-6, 1.0, np.nan_to_num(sd))
        st = raw.static.iloc[idx][["Age", "Height", "Weight"]].to_numpy()
        with np.errstate(all="ignore"):
            self.s_mu = np.nan_to_num(np.nanmean(st, axis=0))
            s_sd = np.nan_to_num(np.nanstd(st, axis=0))
        self.s_sd = np.where(s_sd < 1e-6, 1.0, s_sd)
        return self

    def transform(self, raw: RawData, idx: np.ndarray) -> dict[str, np.ndarray]:
        H = raw.hourly[idx]
        Z = (np.clip(H, self.lo, self.hi) - self.mu) / self.sd            # (n, T, V) with NaN
        mask = ~np.isnan(Z)
        Z0 = np.nan_to_num(Z)
        T = Z.shape[1]
        # last observation carried forward; 0 (= train mean) before the first observation
        last = np.maximum.accumulate(np.where(mask, np.arange(T)[None, :, None], 0), axis=1)
        seen = np.maximum.accumulate(mask, axis=1)
        locf = np.where(seen, np.take_along_axis(Z0, last, axis=1), 0.0)
        x = np.stack([locf, mask.astype(np.float32)], axis=-1).astype(np.float32)   # (n, T, V, 2)

        # per-variable summary features for the feature-concatenation baseline
        cnt = mask.sum(1)
        has = cnt > 0
        mean = np.where(has, Z0.sum(1) / np.maximum(cnt, 1), 0.0)
        mn = np.where(has, np.where(mask, Z, np.inf).min(1), 0.0)
        mx = np.where(has, np.where(mask, Z, -np.inf).max(1), 0.0)
        feats = np.concatenate([mean, mn, mx, locf[:, -1, :], cnt / T], axis=1).astype(np.float32)

        st = raw.static.iloc[idx]
        num = st[["Age", "Height", "Weight"]].to_numpy()
        s_miss = np.isnan(num).astype(np.float32)
        s_num = np.nan_to_num((num - self.s_mu) / self.s_sd).astype(np.float32)
        gender = st["Gender"].fillna(2).to_numpy().astype(np.int64)
        icu = st["ICUType"].fillna(5).to_numpy().astype(np.int64) - 1
        s_cat = np.stack([np.clip(gender, 0, N_GENDER - 1), np.clip(icu, 0, N_ICU - 1)], axis=1)
        s_vec = np.concatenate([s_num, s_miss, np.eye(N_GENDER, dtype=np.float32)[s_cat[:, 0]],
                                np.eye(N_ICU, dtype=np.float32)[s_cat[:, 1]]], axis=1)
        return {
            "x": x, "f": feats, "s_num": s_num, "s_miss": s_miss, "s_cat": s_cat,
            "s_vec": s_vec.astype(np.float32), "y": raw.y[idx],
            # patient-level summaries used only to build the graph (NaN = never measured)
            "var_mean": np.where(has, mean, np.nan).astype(np.float32),
            "static_raw": num.astype(np.float32),
            "static_cat_raw": np.stack([st["Gender"].to_numpy(), st["ICUType"].to_numpy()], 1),
        }


def probe_targets(raw: RawData, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Auxiliary outcomes used ONLY as frozen-embedding probe targets (never as inputs)."""
    oc = raw.outcomes.iloc[idx]
    out = {}
    for col, key in (("Length_of_stay", "los"), ("SAPS-I", "saps"), ("SOFA", "sofa")):
        v = oc[col].to_numpy(np.float64)
        out[key] = np.where(v < 0, np.nan, v)      # -1 marks missing in Outcomes files
    return out


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Download / cache PhysioNet 2012 data")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sets", default="a", help="e.g. a or abc")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rd = load_raw(a.data_dir, tuple(a.sets))
    log.info("N=%d  mortality=%.3f  hourly=%s  observed fraction=%.3f",
             len(rd), rd.y.mean(), rd.hourly.shape, (~np.isnan(rd.hourly)).mean())


if __name__ == "__main__":
    _cli()
