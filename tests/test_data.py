import numpy as np

from tsgf.data import DYNAMIC_VARS, N_HOURS, N_VARS, Preprocessor, RawData, parse_record, stratified_split

RECORD = """Time,Parameter,Value
00:00,RecordID,132539
00:00,Age,54
00:00,Gender,0
00:00,Height,-1
00:00,ICUType,4
00:00,Weight,81.5
00:07,HR,70
00:37,HR,80
01:10,HR,90
00:20,DiasABP,-1
47:59,Temp,36.5
20:00,Weight,82
"""


def test_parse_record_static_and_binning():
    static, hourly = parse_record(RECORD)
    assert hourly.shape == (N_HOURS, N_VARS)
    age, gender, height, icu, weight = static
    assert (age, gender, icu, weight) == (54, 0, 4, 81.5)
    assert np.isnan(height)                                    # -1 -> missing
    hr = DYNAMIC_VARS.index("HR")
    assert hourly[0, hr] == 75 and hourly[1, hr] == 90         # hourly mean
    assert np.isnan(hourly[2, hr])
    assert np.isnan(hourly[0, DYNAMIC_VARS.index("DiasABP")])  # negative = invalid
    assert hourly[47, DYNAMIC_VARS.index("Temp")] == np.float32(36.5)
    # only HR (2 h) and Temp are valid: the -1 DiasABP is dropped and the Weight series is not dynamic
    assert np.isfinite(hourly).sum() == 3


def test_preprocessor_locf_and_mask(synthetic_raw):
    p = Preprocessor().fit(synthetic_raw, np.arange(120))
    d = p.transform(synthetic_raw, np.arange(120, 140))
    assert d["x"].shape == (20, N_HOURS, N_VARS, 2)
    value, mask = d["x"][..., 0], d["x"][..., 1]
    assert set(np.unique(mask)) <= {0.0, 1.0}
    # before the first observation the filled value is exactly 0
    first = np.where(mask.any(1), mask.argmax(1), N_HOURS)
    for i, v in zip(*np.nonzero(first > 0)):
        assert np.all(value[i, : first[i, v], v] == 0)
    # an unobserved step after an observation repeats the previous value
    i, t, v = next((i, t, v) for i in range(20) for v in range(N_VARS) for t in range(1, N_HOURS)
                   if mask[i, t, v] == 0 and mask[i, :t, v].any())
    assert value[i, t, v] == value[i, t - 1, v]
    assert d["f"].shape == (20, 5 * N_VARS) and d["s_vec"].shape == (20, 14)


def test_preprocessor_statistics_use_train_rows_only(synthetic_raw):
    tr = np.arange(100)
    a = Preprocessor().fit(synthetic_raw, tr)
    corrupted = RawData(hourly=synthetic_raw.hourly.copy(), static=synthetic_raw.static.copy(),
                        outcomes=synthetic_raw.outcomes.copy())
    corrupted.hourly[100:] = 1e6                               # change every non-train row
    corrupted.static.loc[100:, "Age"] = 1e6
    b = Preprocessor().fit(corrupted, tr)
    for k in ("lo", "hi", "mu", "sd", "s_mu", "s_sd"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))


def test_stratified_split_is_a_partition():
    y = (np.random.default_rng(1).random(1000) < 0.14).astype(float)
    tr, va, te = stratified_split(y, seed=3, val_frac=0.15, test_frac=0.15)
    assert len(set(tr) | set(va) | set(te)) == 1000
    assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))
    assert abs(len(te) - 150) <= 1 and abs(len(va) - 150) <= 1
    assert abs(y[tr].mean() - y.mean()) < 0.01 and abs(y[te].mean() - y.mean()) < 0.02
    tr2, _, _ = stratified_split(y, seed=3, val_frac=0.15, test_frac=0.15)
    np.testing.assert_array_equal(tr, tr2)                     # deterministic
