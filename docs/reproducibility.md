# Execution and result checks

## 11 November 2026 maintenance verification

The AI-assisted maintenance pass used Python 3.12.16 on Windows, CPU only,
with NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0,
PyTorch 2.15.0+cpu, PyG 2.8.1, and Matplotlib 3.10.8. The audit environment
passed its dependency consistency check.

All 42 tests passed. New regression tests verify that incompatible resumed
protocols, changed experiment definitions, duplicate result keys, missing
configuration records, and misspelled experiment names are rejected. A real
CLI execution on synthetic data verifies that a completed run is skipped,
its evidence is preserved, and a changed epoch budget fails before modifying
that run. Compatible extensions retain earlier seeds and experiment definitions.

`python scripts/verify_results.py` checked each committed 75-row run for complete
experiment/seed coverage, duplicate keys, finite/range-valid core metrics, and
agreement with the six main README AUROC/AUPRC rows for each mode. This verifies
arithmetic, not the original training execution.

A separate one-epoch, seed-0 CPU smoke run downloaded public PhysioNet set-a,
parsed 4,000 stays, and trained `static_mlp`, `concat_mlp`, and `graph_corr_k4`.
The quick configuration was used with two CPU threads and a separate data/output
directory; report generation completed too. These three smoke rows are not
research results and were not substituted for the published tables.

Neither the five-split CPU study nor the GPU study was retrained. The original
CSV, configuration, environment, log, and figure artifacts remain unchanged.
An upstream `torch.jit.script` deprecation warning did not cause a failure.

## Earlier verification record

The project was installed in a fresh Python 3.12.14 virtual environment on
Linux, CPU only. PyTorch was installed from the CPU wheel index, followed by
`pip install -r requirements.txt`. `pip check` found no dependency conflicts,
and all 31 synthetic-data tests passed.

The tested stack was NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn
1.9.1, PyTorch 2.14.1+cpu, PyG 2.8.0.post1, and Matplotlib 3.11.2. Tests used
one BLAS/OpenMP thread. Training used the six PyTorch threads in the config;
other numerical libraries were limited to two threads.

PhysioNet set-a and its outcomes file downloaded successfully. The loader
parsed 4,000 stays. The following check trained every configured model and
graph control, then exercised the reports and figures:

```bash
python -m tsgf.run --config configs/quick.yaml --seeds 0 --epochs 1 --output-dir results/smoke-all
python -m tsgf.report --results results/smoke-all
```

It completed all 15 experiment rows, with no duplicate `(experiment, seed)`
keys. This one-epoch run checks execution; its metrics are not used to interpret
the research question or replace the committed results.

The committed quick and full CSVs each contain 75 rows without duplicate keys.
Recomputing their per-model AUROC means and sample SDs reproduced the headline
values in the README. The five-split, 30-epoch CPU experiment and the full GPU
experiment were not retrained in this check. Their outputs, logs, configurations,
and original environment records remain in `results/quick/` and `results/full/`.
