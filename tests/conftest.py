import numpy as np
import pandas as pd
import pytest

from tsgf.data import N_HOURS, N_VARS, OUTCOME_COLS, STATIC_COLS, RawData


@pytest.fixture(scope="session")
def synthetic_raw() -> RawData:
    """Small random dataset with the same layout as PhysioNet 2012 (no download needed)."""
    rng = np.random.default_rng(0)
    n = 240
    hourly = rng.normal(size=(n, N_HOURS, N_VARS)).astype(np.float32) * 10 + 50
    hourly[rng.random(hourly.shape) < 0.8] = np.nan
    age = rng.uniform(20, 90, n)
    risk = (age - 55) / 20 + np.nan_to_num(np.nanmean(hourly[:, :, 0], axis=1) - 50) / 5
    y = (risk + rng.normal(size=n) > 0.8).astype(float)
    static = pd.DataFrame({"Age": age, "Gender": rng.integers(0, 2, n).astype(float),
                           "Height": rng.normal(170, 10, n), "ICUType": rng.integers(1, 5, n).astype(float),
                           "Weight": rng.normal(80, 15, n)})[STATIC_COLS]
    static.loc[rng.random(n) < 0.3, "Height"] = np.nan
    oc = pd.DataFrame({"RecordID": np.arange(n), "SAPS-I": rng.integers(0, 30, n).astype(float),
                       "SOFA": rng.integers(0, 15, n).astype(float),
                       "Length_of_stay": rng.integers(2, 30, n).astype(float),
                       "In-hospital_death": y})[OUTCOME_COLS]
    return RawData(hourly=hourly, static=static, outcomes=oc)
