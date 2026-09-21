# Learning roadmap

A reading and building order from this repository towards graph-structured diffusion for mixed
static + temporal data. Titles, identifiers and listed venues were checked against the linked primary records;
formal writing should still use the citation exported by the publisher or paper page.

## Stage 0 — understand what is in this repo (1–2 days)

- Run the tests, read `tsgf/graph.py` and `tsgf/models.py` (≈ 350 lines together).
- Reproduce one row of the results table with a different seed; change `k_var` and watch which edges move.
- Exercise: implement Pearson instead of Spearman in `association_matrices`; does the informed graph change?

## Stage 1 — graph neural networks

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017), [arXiv:1609.02907](https://arxiv.org/abs/1609.02907).
- Veličković et al., *Graph Attention Networks* (ICLR 2018), [arXiv:1710.10903](https://arxiv.org/abs/1710.10903).
- Brody, Alon, Yahav, *How Attentive are Graph Attention Networks?* (ICLR 2022), [arXiv:2105.14491](https://arxiv.org/abs/2105.14491) — the GATv2 used here.
- Gilmer et al., *Neural Message Passing for Quantum Chemistry* (ICML 2017), [arXiv:1704.01212](https://arxiv.org/abs/1704.01212) — the message-passing view.
- Build: swap `GATv2Conv` for `GCNConv` with edge weights = association strength; compare.

## Stage 2 — static + temporal fusion and irregular time series

- Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data* (NeurIPS 2021), [arXiv:2106.11959](https://arxiv.org/abs/2106.11959) — per-feature tokens
  (the static tokeniser here is a simplified version of the idea) and why a strong non-neural baseline is needed.
- Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports 8, 6085 (2018),
  doi:10.1038/s41598-018-24271-9 (GRU-D) — masks and time-since-last-observation.
- Horn et al., *Set Functions for Time Series* (ICML 2020), [arXiv:1909.12064](https://arxiv.org/abs/1909.12064) — evaluates on the same PhysioNet 2012 task.
- Zhang et al., *Graph-Guided Network for Irregularly Sampled Multivariate Time Series* (ICLR 2022), [arXiv:2110.05357](https://arxiv.org/abs/2110.05357) —
  the closest prior work: a graph over sensors for irregular series (no static modality as nodes, learned rather than fixed structure).
- Build: add a time-since-last-observation channel; replace the GRU with a small 1D-CNN and check that the *fusion* conclusions survive.

## Stage 3 — generative models

- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models* (NeurIPS 2020), [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
- Tashiro et al., *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation* (NeurIPS 2021), [arXiv:2107.03502](https://arxiv.org/abs/2107.03502)
  — conditional diffusion for irregular clinical time series (uses PhysioNet 2012).
- Kotelnikov et al., *TabDDPM: Modelling Tabular Data with Diffusion Models* (ICML 2023), [arXiv:2209.15421](https://arxiv.org/abs/2209.15421).
- Jo, Lee, Hwang, *Score-based Generative Modeling of Graphs via the System of SDEs* (ICML 2022), [arXiv:2202.02514](https://arxiv.org/abs/2202.02514).
- Vignac et al., *DiGress: Discrete Denoising Diffusion for Graph Generation* (ICLR 2023), [arXiv:2209.14734](https://arxiv.org/abs/2209.14734).
- Build (the eventual thesis question): condition a denoiser for the temporal nodes on the static nodes through the *same*
  fixed edge set and compare with unconditioned/concatenated conditioning — using the control graphs defined here
  (`none`, `random`, `permuted`) as the ablation. Only attempt it if the discriminative study shows that structure matters.

## Stage 4 — evaluation habits worth building

- Report paired, per-split differences; keep every control that has the same capacity as the method.
- Keep a written list of everything that touches labels or test data; add a test for each rule (see `tests/test_data.py`).

## Dataset references

- Silva I, Moody G, Scott DJ, Celi LA, Mark RG. *Predicting In-Hospital Mortality of ICU Patients: The
  PhysioNet/Computing in Cardiology Challenge 2012.* Computing in Cardiology 2012; 39:245–248.
  (Title and authors as listed on the dataset page; the page asks users to cite this paper.)
- Pollard T et al. *PhysioNet as a global platform for biomedical research.* Nature Health (2026),
  <https://doi.org/10.1038/s44360-026-00096-z> (the current standard PhysioNet citation listed on the dataset page).
- Dataset: <https://physionet.org/content/challenge-2012/1.0.0/>; files are under the
  [Open Data Commons Attribution License v1.0](https://physionet.org/content/challenge-2012/view-license/1.0.0/).
