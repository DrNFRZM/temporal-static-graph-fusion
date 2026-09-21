# Problem formulation

Notation follows the code: `T = 48` hourly steps, `V = 36` dynamic variables, `S = 5` static descriptors,
`d = d_node`, `h = d_hidden`. Concrete values are in `tsgf/data.py`.

## 1. Data

A sample is one ICU stay `i` with two modalities and a binary label.

**Static modality.** Admission descriptors `s_i = (Age, Height, Weight, Gender, ICUType)`.

- Three numeric descriptors `s_i^num ∈ (ℝ ∪ {∅})^3` (∅ = missing; Height/Weight are often missing).
- Two categorical descriptors: Gender ∈ {F, M, missing}, ICUType ∈ {1,…,4, unknown}.
- Static information is *thin* (5 fields). That is a property of this dataset and is discussed in
  `design_decisions.md`.

**Temporal modality.** Irregularly sampled measurements of `V` variables during the first 48 h,
aggregated to hourly means: `X_i ∈ (ℝ ∪ {∅})^{T×V}` with observation mask
`M_i ∈ {0,1}^{T×V}`, `M_i[t,v] = 1` iff variable `v` was measured in hour `t`.
Only ~19 % of the cells are observed, and how often a variable is measured is itself informative.

**Label.** `y_i ∈ {0,1}`: in-hospital death.

**Preprocessing** `φ` (statistics from the training split only): winsorise at the 0.5 / 99.5 % quantiles,
z-score per variable, carry the last observation forward, and pass the mask along:

```
x̃_i[t,v] = ( LOCF(z(X_i))[t,v] , M_i[t,v] )   ∈ ℝ^2 ,      x̃_i ∈ ℝ^{T×V×2}
```

## 2. Encoders

**Temporal encoder** `f_t` — one GRU with weights shared across variables, applied to each variable's own
2-channel series, plus a learned variable embedding `e_v ∈ ℝ^d`:

```
h_v = LayerNorm( GRU( x̃[:, v, :] )_{t=T}  +  e_v ) ∈ ℝ^d ,      H_t = [h_1 … h_V] ∈ ℝ^{V×d}
```

**Static encoder** `f_s` — a tokeniser, one `d`-dim vector per descriptor
(numeric: `w_k · s_k + b_k` plus a learned missing-token; categorical: an embedding lookup):

```
u_k = LayerNorm( token_k(s_k) ) ∈ ℝ^d ,      H_s = [u_1 … u_S] ∈ ℝ^{S×d}
```

## 3. Graph

A fixed **node set** shared by all patients, `𝒱 = 𝒱_t ∪ 𝒱_s`, `|𝒱_t| = V`, `|𝒱_s| = S`, `N = V + S = 41`.
Node `n`'s feature for patient `i` is a row of `H_i = [H_t ; H_s] ∈ ℝ^{N×d}`.

A **graph** `G = (𝒱, ℰ)` is a fixed, dataset-level edge set (the same for every patient). The informed
graph `G_A` is built from association strengths `A` estimated on training patients:

- per-patient summary `m_i[v]` = mean of the z-scored observed values of variable `v` (NaN if never measured);
- `A_vv[a,b] = |ρ_Spearman(m[a], m[b])|` (pairwise-complete, ≥ 30 patients);
- `A_sv[k,v] = |ρ_Spearman(descriptor_k, m[v])|` (categorical descriptors: max over level indicators);
- `ℰ_vv` = symmetrised top-`k_var` neighbours of every variable node (default 4);
- `ℰ_sv` = for each static node, its top-`k_static` variables, in both directions (default 6).

Semantics: *an edge says “these two measurements co-vary across patients”* (variable–variable), or
*“this admission descriptor is associated with the level of this measurement”* (static–variable).
It is a **statistical prior**, not a causal or physiological claim, and it never sees the label.

Control graphs (same node set, different `ℰ`): `none`, `full`, `random` (same number of edges of each type,
random endpoints), `permuted` (informed topology, node identities shuffled within type, so the degree
sequence is preserved), `varvar_only`, `cross_only`.

## 4. Fusion and prediction

`L = 2` pre-norm residual GATv2 layers over `G` (`H^{l+1} = H^l + Dropout(GELU(GATv2(LN(H^l), ℰ)))`,
4 heads, self-loops added by PyG), applied to all patients in a mini-batch as one disjoint union of copies of `G`.

Readout (default `flatten`): `z_i = MLP( vec(H_i^{L}) ) ∈ ℝ^h`; prediction `p_i = σ(w^T z_i + b)`.
`z_i` is the **joint embedding** evaluated by the probes. Training loss: binary cross-entropy.

## 5. Baselines as special cases

| model | what it isolates |
|---|---|
| `static_mlp`, `temporal_only` | single-modality ablations |
| `concat_mlp` | *feature* concatenation of hand-crafted summaries (mean/min/max/last/frac. observed per variable) and the static vector |
| `late_fusion` | same temporal encoder as the graph model, flattened; static MLP; embeddings concatenated |
| `graph_none` | the graph model with `ℰ = ∅`: same tokenisation/readout and GATv2 self-updates, but no information exchange between different nodes |
| `graph_full` | `ℰ` = all pairs (≈ a small transformer over 41 tokens) |
| `graph_random`, `graph_permuted` | same edge budget / degree sequence, wrong semantics |

## 6. Hypotheses and how they can fail

- **H1 (fusion).** Graph fusion gives a better test AUROC/AUPRC than `concat_mlp` and `late_fusion`.
  *Fails if* the paired difference is ≈ 0 or negative across splits.
- **H2 (structure).** The informed graph beats controls with the same capacity (`none`, `random`, `permuted`).
  *Fails if* controls match it; then any gain in H1 comes from tokenisation/capacity, not from graph structure.
- **H3 (embedding).** Frozen joint embeddings give better probe scores (mortality, LOS ≥ median, SAPS-I R²)
  than the baselines' embeddings.
- **H4 (cross-modal edges).** Removing static–variable edges (`varvar_only`) hurts more than removing
  variable–variable edges (`cross_only`) — i.e. the *bridge* between modalities is what the graph adds.

H1–H4 are the hypotheses used to organise the reported comparisons. They were not externally or
timestamp-preregistered, so the repository does not present them as a formal preregistration. Whether they held
is reported in `README.md`.

## 7. Relation to a diffusion / generative model (out of scope here)

If the graph is later used as a structural prior for a generative model, its edges would constrain which
static and temporal quantities condition each other in the denoiser. This repository only asks the
prerequisite question: *does such a graph help a discriminative joint representation at all?*
