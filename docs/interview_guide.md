# Interview guide

Numbers below are **quick mode** (set-a, N = 4,000, mean ± sample SD across 5 random splits) from
`results/quick/summary.md` unless stated. Full mode (12,000 stays, GPU, 5 splits) is in `results/full/summary.md` and `README.md`;
it leads to the same qualitative conclusion with a smaller observed paired spread (SD ≈ 0.003–0.007). The
splits overlap and are not independent, so these SDs are descriptive—not confidence intervals.

## 1. Elevator explanation (30 s)

I asked whether representing static patient descriptors and irregular physiological time series as nodes of one
graph gives a better joint representation than concatenating them. I built the graph, three baselines and
a set of control graphs (isolated nodes, random edges, shuffled node identities, all edges) on the PhysioNet 2012
ICU data, with identical training and 5 repeated splits. Fusion was better in every recorded split
(temporal-only 0.816 → fused ≈ 0.83–0.84 AUROC), but **the graph structure itself did not help**: a model with
no inter-node edges scored the same as the informed graph. That is a negative result on the graph-as-prior idea at this scale, and it
tells me what a diffusion-model project would have to show before the graph earns its place.

## 2. Technical explanation

Each stay has 5 static descriptors and a 48 h × 36 variable hourly tensor (81 % missing). A shared GRU encodes every
variable's own (LOCF value, observed flag) series into a vector; a tokeniser turns each static descriptor into a
vector. The 41 vectors are nodes of a fixed graph whose edges are the strongest |Spearman| associations between
variables (and between descriptors and variables), estimated on the training split only. Two residual GATv2 layers
pass messages; the node states are flattened into an MLP whose output is the joint embedding; a linear head predicts mortality.
Controls keep the architecture fixed while changing inter-node connectivity. In `graph_none`, PyG self-loops
still apply a learned update to each node, but different nodes cannot exchange information. Embeddings are
frozen and probed with linear/kNN classifiers and a ridge regressor.

## 3. Node and edge interpretation

- **Temporal node** `v`: "what variable `v` did over the 48 h", a learned summary of its (value, observed) series.
- **Static node** `k`: "admission descriptor `k`" (Age, Height, Weight, Gender, ICU type), a learned token; missing values map to a learned missing-token.
- **Variable–variable edge**: the two variables' patient-level means co-vary across the training population (e.g. `NIDiasABP–NIMAP`,
  `BUN–Creatinine`, `ALT–AST`; these are the strongest edges the pipeline found).
- **Static–variable edge**: the descriptor is associated with the level of the measurement.
- Edges are a *population statistic*, not a causal or physiological graph, are undirected (stored as two directed edges),
  identical for every patient, and computed without labels. Per-patient variation comes from node features and GATv2 attention.

## 4. Tensor shapes (batch size B; quick config `d=32`, `h=64`; informed graph `k_var=4, k_static=6`)

| step | shape |
|---|---|
| temporal input `x` | `(B, 48, 36, 2)` — LOCF z-value, observed flag |
| reshaped for the shared GRU | `(B·36, 48, 2)` |
| GRU final hidden → per variable | `(B, 36, 32)` (+ variable embedding, LayerNorm) |
| static tokens | `(B, 5, 32)` from `s_num (B,3)`, `s_miss (B,3)`, `s_cat (B,2)` |
| node features `H` | `(B, 41, 32)` → flattened `(B·41, 32)` for PyG |
| edge index (shared graph, replicated with offsets) | `(2, B·254)` (254 directed edges, 60 of them cross-modal) |
| GATv2 layer (4 heads × 8) | `(B·41, 32)` → residual → `(B, 41, 32)` |
| flatten readout | `(B, 1312)` → MLP → embedding `z` `(B, 64)` |
| head | `(B, 1)` logit |

Baselines: `concat_mlp` input `(B, 180 + 14)` (5 summary stats × 36 variables + static vector); `late_fusion`
flattens the same `(B, 36, 32)` temporal states to `(B, 1152)`.

## 5. Why graph fusion rather than concatenation?

The motivation is structural: concatenation lets the network learn any interaction but gives it no prior over *which*
static and temporal quantities should interact, and it mixes modalities only at the end. A graph makes the interaction
pattern an explicit, inspectable object, which is what a later generative model could condition on. In this study the
empirical answer at this scale is that it does not buy predictive performance: `graph_corr_k4` 0.838 ± 0.012 vs
`concat_mlp` 0.839 ± 0.014 and `graph_none` 0.838 ± 0.016. What the graph *does* offer here is a testable structure, and the controls
show the network does not need it.

## 6. Why a per-variable GRU?

To get one vector per variable so the temporal modality can be nodes; weights are shared, so the parameter count stays small
and rare variables borrow statistical strength. GRU rather than CNN/Transformer because the series are short, sparse and
LOCF-filled with an explicit mask, so recurrence tracks "how stale is this value" naturally. I did not compare encoders; this is a limitation.

## 7. Potential leakage — what I guarded against and what remains

Guarded (with tests where possible):
- Winsorising bounds, means/stds, static normalisation: **train-only** (`test_preprocessor_statistics_use_train_rows_only`).
- The graph associations are computed from **training patients only**, per seed; edges never use labels.
- SAPS-I, SOFA, length of stay, survival days exist in the Outcomes files; **never inputs** — only probe targets.
- One patient = one record; the stratified split is over stays, so no within-patient leakage inside this dataset's release.
- Batched message passing does not mix patients (`test_samples_do_not_interact`).
- Test metrics are computed once from best-validation weights.

Remaining / not addressed:
- **Missingness is informative** (which labs were ordered reflects clinician concern). It is a legitimate predictor but not "physiology alone".
- Random splits are not the challenge's A/B/C split; repeated draws from the same 4,000 stays share the same hospitals/era, so the variance is optimistic.
- The repeated splits overlap heavily; five seeds are not five independent samples, and their SD cannot be read as a confidence interval.
- 12,000 stays come from a small set of ICUs; no external validation.
- Model selection on validation AUROC with a small validation set (600 stays) is noisy.

## 8. Limitations

- Negative result at n = 5 splits: paired SD of differences ≈ 0.01 (quick) / 0.003–0.007 (full) AUROC, so effects of a few thousandths are not resolvable.
- The static modality has only 5 fields; a richer static side (genomics, EHR history) could behave differently.
- One dataset, one target, one temporal encoder, untuned shared hyper-parameters, parameter counts not matched
  between baselines and graph models (`concat_mlp` has 17k parameters vs 98k).
- The edge prior is one population-level graph; it cannot express patient-specific structure.
- Probes are on supervised embeddings, so they say what a supervised model retained, not label-free representation quality.
- ICU data are a benchmark here; nothing is clinically validated.
- Demographic fields are present, but no subgroup or fairness analysis was performed.

## 9. Fifteen difficult questions

**1. Your graph model equals the isolated-node control. Isn't the project a failure?**
It is a negative result on H2, obtained with controls that make it interpretable. The controls answer the question the project posed:
at this scale, the fusion benefit (temporal-only 0.816 → ≈ 0.83–0.84) comes from *having* both modalities and a reasonable readout, not from message passing over an association graph.
That is exactly the evidence needed before spending a thesis on graph priors for diffusion.

**2. How do you know the null is not just low power?**
I do not. In quick mode the paired SD is ≈ 0.01 AUROC. With 12,000 stays the observed paired SD is
≈ 0.003–0.007 and the informed graph remains within ±0.003 of every control (largest: +0.003 over the
all-pairs graph, 4/5 seeds). That is stronger descriptive evidence that there is no practically large advantage
under this setup, but the five overlapping splits do not establish equivalence or a confidence bound. A small
effect remains unresolved.

**3. Why Spearman on patient-level means as the edge weight? That ignores time.**
It is the cheapest label-free dependence measure computable identically for variable–variable and static–variable pairs with missing data. It ignores temporal
lead–lag structure, which is a real limitation; alternatives are lagged cross-correlation, Granger-type statistics, or a learned adjacency.
The sanity check that supports it is that top edges are physiologically sensible (blood-pressure variants, BUN–creatinine, ALT–AST).

**4. Random and permuted graphs performing as well as the informed one could mean the GNN ignores edges. Did you check?**
The `none` vs `informed` vs `random` triple is the relevant performance check: accuracy is insensitive to all
three, so the runs provide no evidence that the network exploited the inter-node edge set. That is not proof
that every learned attention coefficient is unused. I did not inspect attention weights; that would be the next
diagnostic (or an interchange test of the learned attention).
An additional caveat is the 2-layer residual design: information also reaches the readout directly (skip connections + flatten), so the graph is easy to ignore.

**5. Why not a fully learned graph?**
It would confound "graph helps" with "optimiser found a graph". A fixed prior gives a clean structural test. A learned adjacency initialised at the prior is a natural follow-up.

**6. Concatenation with hand-crafted summaries matches the neural models. What does that say?**
That on 4,000 stays with 81 % missing cells, summary statistics carry most of the usable signal and the shared GRU adds no advantage (`temporal_only` 0.816 < `concat_mlp` 0.839, and gradient boosting on the same features 0.841). Deep temporal encoders show their value with more data or richer dynamics.

**7. The parameter counts differ (17k vs 98k). Is the comparison fair?**
Not perfectly, and I report it. The structure comparison (`none/random/permuted/full` vs informed) is capacity-matched (identical parameter counts). The `mean` readout variant has 18k parameters and scores 0.841 ± 0.021, which suggests capacity is not what limits the results.

**8. Why AUROC as primary metric, and what about calibration?**
AUROC is threshold-free and standard for this benchmark; AUPRC is reported because the prevalence is 14 %; Brier is in the CSV. I did not analyse calibration or decision curves, so no deployment claims.

**9. Why not use the official challenge splits/score?**
I wanted repeated splits to estimate variance, and the challenge's event-1 score is a threshold metric on a separate hidden set. So my numbers are not comparable to challenge leaderboards or to papers that use fixed splits.

**10. Isn't LOCF plus mask a form of leakage or a shortcut?**
LOCF uses only past values, so it is causal within the 48 h window. The mask is a shortcut only if you want "physiology only": ordering frequency is predictive because clinicians order tests for sicker patients. I kept it deliberately and state it.

**11. What would you change if static information were rich?**
Static nodes would then have many more edges to compete for and the cross-modal edge type would matter more. H4 (cross-modal edges matter) was not supported here (`varvar_only` 0.842 vs `cross_only` 0.837, within noise); with only 5 descriptors, one would not expect it to be.

**12. How do embedding probes relate to "representation quality"?**
They test what a frozen supervised embedding retains: mortality (re-fit), an unseen outcome (LOS ≥ median) and a severity score (SAPS-I R²). Results were similar across neural models, except that `concat_mlp` retains more SAPS-I information (R² 0.50 vs 0.40 for the graph model). The probes are not label-free, so they cannot support claims about self-supervised representations.

**13. Why GATv2 rather than GCN or a transformer?**
GATv2 lets the same fixed topology be weighted per patient. A fully connected GATv2 is roughly a small transformer over 41 tokens, which is the `graph_full` control; it also matches. A GCN with edge weights would be the natural alternative and is in the roadmap.

**14. How does this connect to diffusion?**
It is the discriminative precondition: if a fixed association graph does not improve a supervised joint embedding, a diffusion model must justify the graph differently, for instance through sample quality, conditional generation, or sample efficiency. The controls used here (`none`, `random`, `permuted`) carry over directly as ablations for a conditional denoiser. This repository makes no claim about generation.

**15. What is the single most important next experiment?**
Test where structure *should* matter: reduce the training set size (e.g. 5–20 % of the stays) and check whether the informed graph beats `none/random/permuted` in the low-data regime, where a prior has the most room to help. If the ordering is still flat, drop the graph-prior hypothesis for this data; if not, that is the regime for the diffusion study.
