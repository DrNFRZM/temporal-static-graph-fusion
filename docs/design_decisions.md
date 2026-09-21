# Design decisions

Each entry: what was chosen, what was considered, and what it costs.

## Dataset: PhysioNet/CinC Challenge 2012

- **Chosen.** ICU stays of ≥ 48 h from the PhysioNet/Computing in Cardiology Challenge 2012
  (Silva et al., 2012). Static descriptors at admission + 36 irregularly sampled physiological/lab variables
  over 48 h + in-hospital mortality. Open access, Open Data Commons Attribution License v1.0, downloadable
  without login: <https://physionet.org/content/challenge-2012/1.0.0/>. The code downloads only
  `set-{a,b,c}.tar.gz` and `Outcomes-{a,b,c}.txt` (≈ 7 MB each).
  I checked that the files download anonymously and that set-a parses to 4,000 records.
- **Why this one.** The fusion question only makes sense if *both* modalities carry signal. Here static-only
  gets test AUROC ≈ 0.66 and temporal-only ≈ 0.82 (quick mode, `results/quick/summary.md`), so there is something to fuse.
- **Considered and rejected.**
  - *Wearable activity datasets* (MotionSense, PAMAP2, UCI-HAR, MHEALTH): open and easy, but the "static" side is
    subject-level metadata for 9–30 people, which cannot be split without leaking subject identity and has
    almost no predictive relation to the activity label.
  - *CAPTURE-24* (151 participants, wrist accelerometry): a better static/temporal story but ≈ GB-scale download.
  - *MIMIC-III/IV, eICU*: credentialed access; not frictionless, so not used.
- **Framing.** ICU data are used as a *methodological benchmark* for static + irregular time-series fusion,
  not as a clinical study. No clinical claims are made, and nothing here is validated for care.
  Demographic descriptors are present, but no subgroup or fairness analysis is performed. The setting is a
  common benchmark in the irregular-time-series literature, not a clinical or fairness evaluation.

## Task and split

- Target: in-hospital mortality (13.9 % positive in set-a).
- Every seed draws a **new stratified 70/15/15 split** *and* a new initialisation. This measures
  split + init variation together. Because the five splits overlap, the reported SDs are descriptive rather
  than confidence intervals or estimates from independent samples.
  I did not use the challenge's own A/B/C split so that all 12,000 stays can be used in full mode.
- Early stopping on validation AUROC; test metrics are computed once, from the best-validation weights.

## Preprocessing

- Hourly mean binning, negatives treated as invalid, winsorise + z-score with **train-only** statistics,
  last-observation-carried-forward plus an explicit observed-mask channel.
- The mask is kept on purpose: *when* something is measured is a real signal in ICU data. It is also a
  reason absolute AUROCs here should not be read as "physiology only" (see the leakage section of the interview guide).
- Weight is static only (its time-series copy in the raw files is dropped) so no descriptor appears in both modalities.

## Temporal encoder: one shared per-variable GRU

- **Chosen.** A single GRU whose weights are shared across variables and applied to each variable's own
  (value, mask) series, plus a learned variable embedding. Output: one vector per variable.
- **Why per-variable.** It makes the temporal modality *decomposable into nodes*, which is what a graph needs.
  A multivariate GRU would give one vector for the whole stay and leave nothing to connect the static side to.
- **Why GRU rather than CNN / Transformer.** Series are short (48) and sparse; the LOCF+mask input makes a GRU's
  recurrence a natural "time since last measurement" tracker; parameter count is small. A 1D-CNN would be faster on
  CPU; a Transformer over 48 steps × 36 variables would add cost without a clear benefit at this size. I did *not*
  compare encoders — the study is about fusion, and the encoder is held fixed across all neural models.
- **Cost.** A shared GRU with a small hidden size is a bottleneck for variables with very different dynamics.

## Graph: what a node and an edge mean

- One node per **variable** (its temporal summary vector) and one per **static descriptor** (its token).
  The node set is fixed and shared, so a patient is a *labelled* graph where the labels are the modality fields.
- Edges = training-set association strength (see `problem_formulation.md`). Chosen because it is
  (a) label-free, so it cannot leak the target, (b) computable identically for both modality pairs,
  (c) inspectable — the strongest learned edges are e.g. `NIDiasABP–NIMAP`, `DiasABP–MAP`, `ALT–AST`, `BUN–Creatinine`,
  which agree with clinical intuition, a cheap face-validity check.
- **Rejected alternatives.** (i) One node per hour (a "temporal graph"): 48 nodes per stay, but the static side
  has no natural connection to hours. (ii) Per-patient graphs from within-stay correlations: 48 noisy points
  per pair. (iii) Learned adjacency: interesting, but confounds "does a graph help" with "did the optimiser find one";
  a fixed prior makes the structure test clean. (iv) A knowledge graph from clinical ontologies: not frictionless.
- **Cost.** The prior is one graph for everyone, so it can encode population co-variation but not patient-specific
  structure. Attention in GATv2 supplies per-patient re-weighting of the fixed edges.

## GNN layer and readout

- GATv2 (PyG `GATv2Conv`), 2 layers, pre-norm residual. Attention lets the same topology be used differently
  per patient. PyG self-loops remain enabled, so an empty inter-node graph degrades to isolated per-node
  transformations plus the residual path; information cannot move between different nodes.
- Readout `flatten` keeps node identity (fixed node set), matching how `temporal_only` reads its variable vectors;
  a `mean` readout is included as one sensitivity row because mean pooling is the usual GNN default and can bottleneck.

## Baselines

- `concat_mlp` uses hand-crafted summary features rather than the learned temporal encoder — this is what
  "simple feature concatenation" usually means and is a strong baseline on this data.
- `late_fusion` shares the graph model's temporal encoder exactly, so `late_fusion` vs `graph_*` differ only in
  tokenisation + message passing + readout.
- `graph_none` closes the gap between them: it has the graph model's tokenisation, GATv2 self-updates and
  readout but no inter-node edges.
- `gbm_concat` (HistGradientBoosting on the concat features) is a non-neural reference, not part of the fusion comparison.
- **Capacity is not matched exactly** (parameter counts are printed in the results table). `graph_none / random /
  permuted / full` share the informed graph's parameter count, so the structure comparison is capacity-controlled;
  the comparison to `concat_mlp` is not, and should be read with that in mind.

## Embedding-quality probes

Frozen embeddings → logistic regression / kNN for mortality, logistic for LOS ≥ train-median, ridge for SAPS-I.
SAPS-I, SOFA and LOS are read from the Outcomes files and **used only as probe targets**, never as inputs. They are
severity scores computed from the same 48 h of data, so feeding them in would be target leakage.
Caveat: the encoders are trained supervised on mortality, so these probes measure *what the supervised
representation retains*, not label-free representation quality.

## What is deliberately not here

- **No diffusion / reconstruction experiment.** A reconstruction or masked-imputation task would add a second
  objective and a second set of design choices; the question posed here (does graph structure help joint
  representations?) is answerable without it, and adding it would blur the attribution.
- **No per-model hyperparameter tuning.** All neural models share hidden sizes, dropout, optimiser and epochs.
  The recorded runs use one common setting rather than a search tailored to each model. Larger effects for any
  one model could be hidden by that choice.
- No test-time augmentation, ensembling, pretraining, or class re-weighting.
