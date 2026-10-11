# Execution and result checks

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
