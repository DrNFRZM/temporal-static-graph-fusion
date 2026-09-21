# Experiment plan

Hypotheses H1–H4 are defined in `problem_formulation.md §6`. This file lists what is run, how it is
read, and what is *not* claimed.

## Protocol

| item | quick (CPU) | full (GPU) |
|---|---|---|
| config | `configs/quick.yaml` | `configs/full.yaml` |
| data | set-a, N = 4,000 | sets a+b+c, N = 12,000 |
| model | `d_node=32`, `d_hidden=64` | `d_node=64`, `d_hidden=128` |
| training | ≤ 30 epochs, patience 6, batch 128, lr 2e-3 | ≤ 60 epochs, patience 10, batch 256, lr 1e-3 |
| repeats | 5 seeds | 5 seeds |

- Each seed = new stratified 70/15/15 split + new weight initialisation. Preprocessing statistics,
  graph associations and the random/permuted control graphs are all (re)computed per seed from the training split only.
- Metrics on the test split: AUROC (primary), AUPRC, Brier. Model selection: validation AUROC.
- Summaries: mean ± sample SD over seeds; **paired** differences (same split and seed) with the number of
  seeds in which A > B. The five random splits overlap and are not independent samples, so the SDs are
  descriptive, not confidence intervals. No p-values or statistical-equivalence claims are reported.

## Experiments

| id | name | purpose |
|---|---|---|
| E1 | `static_mlp` | static-only ablation |
| E2 | `temporal_only` | temporal-only ablation (same encoder as the graph model) |
| E3 | `concat_mlp` | **baseline 1**: feature concatenation |
| E4 | `late_fusion` | **baseline 2**: temporal encoder + static MLP, embeddings concatenated |
| E5 | `graph_corr_k4` | **graph fusion** (informed graph, `k_var=4`, `k_static=6`) |
| E6 | `graph_none` | E5 without inter-node edges — isolates connectivity while retaining self-updates and readout |
| E7 | `graph_random` | same edge counts per type, random endpoints |
| E8 | `graph_permuted` | same topology and degrees, node identities shuffled |
| E9 | `graph_full` | all pairs (no prior) |
| E10 | `graph_varvar_only`, `graph_cross_only` | which edge type matters (H4) |
| E11 | `graph_corr_k2`, `graph_corr_k8` | density sensitivity |
| E12 | `graph_meanpool` | readout sensitivity |
| ref | `gbm_concat` | non-neural reference on the E3 features |

Embedding probes (all neural models): logistic-regression and kNN AUROC on mortality, logistic AUROC for
LOS ≥ median, ridge R² for SAPS-I — fit on train embeddings, scored on test embeddings.

## Interpretation criteria

These criteria organise the comparisons but were not externally or timestamp-preregistered.

- **H1 supported** if `graph_corr_k4` − `late_fusion` and − `concat_mlp` in test AUROC are positive in most seeds
  and their mean exceeds the seed SD of the difference.
- **H2 supported** only if `graph_corr_k4` beats *all* of `none`, `random`, `permuted` in the same sense.
  If `graph_none ≈ graph_corr_k4`, the graph structure is not what helps. If `permuted ≈ corr_k4`, the topology
  matters but not the semantics of which nodes are joined, and vice versa for `random`.
- `graph_full` is a control on *sparsity*: if it matches `corr_k4`, the sparse prior buys nothing over letting attention decide.
- **H4** is read from `varvar_only` vs `cross_only` relative to `corr_k4`.
- Probe results are secondary; a model can lead on AUROC and lose on probes, and both are reported.

## Threats to validity (tracked, not solved)

- Parameter counts differ between the baselines and the graph model (reported in the results table).
- The static modality has 5 fields, which limits how much any static–temporal interaction can help.
- One dataset, one target; no external validation set (the 12,000 stays come from the same challenge).
- The temporal encoder is fixed; conclusions are about fusion *given* that encoder.
- Random splits, not the official challenge split, so numbers are not comparable to challenge scores.
- Repeated splits overlap substantially, so their seed-to-seed spread is not an independent-sample uncertainty estimate.
- Shared, untuned hyper-parameters (see `design_decisions.md`).

## Commands

```bash
python -m tsgf.data --sets a                    # download + cache (optional; run does it lazily)
python -m tsgf.run --config configs/quick.yaml  # CPU, all experiments, resumable
python -m tsgf.report --results results/quick   # tables + figures
python -m tsgf.run --config configs/full.yaml   # GPU
python -m tsgf.run --config configs/quick.yaml --seeds 0 --epochs 2 --only static_mlp,graph_corr_k4 --output-dir results/smoke   # sanity check
```

## Status

- Quick mode: executed (5 seeds × 15 experiments, CPU); results in `results/quick/` and `README.md`.
- Full mode: executed (5 seeds × 15 experiments, RTX 2060); results in `results/full/` and `README.md`.
- Outcome: H1–H4 not supported in either mode (see `README.md`).
