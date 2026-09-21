"""Summarise results.csv into tables (summary.md) and figures:  python -m tsgf.report --results results/quick"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAIN = ["static_mlp", "temporal_only", "concat_mlp", "late_fusion", "graph_corr_k4", "gbm_concat"]
SENS = ["graph_corr_k2", "graph_corr_k4", "graph_corr_k8", "graph_varvar_only", "graph_cross_only",
        "graph_meanpool", "graph_full", "graph_none", "graph_permuted", "graph_random"]
INFORMED = {"graph_corr_k2", "graph_corr_k4", "graph_corr_k8", "graph_varvar_only",
            "graph_cross_only", "graph_meanpool"}
CONTROLS = {"graph_full", "graph_none", "graph_permuted", "graph_random"}
PAIRS = [("graph_corr_k4", "concat_mlp"), ("graph_corr_k4", "late_fusion"),
         ("graph_corr_k4", "graph_none"), ("graph_corr_k4", "graph_permuted"),
         ("graph_corr_k4", "graph_random"), ("graph_corr_k4", "graph_full"),
         ("late_fusion", "temporal_only")]
METRICS = ["test_auroc", "test_auprc", "test_brier", "probe_lr_auroc", "probe_knn_auroc",
           "probe_los_auroc", "probe_saps_r2"]

# colours: neutral = baselines, blue = informed graph, orange = structure controls (see README)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
C_BASE, C_GRAPH, C_CTRL = "#898781", "#2a78d6", "#eb6834"


def _color(name: str) -> str:
    return C_GRAPH if name in INFORMED else C_CTRL if name in CONTROLS else C_BASE


def _fmt(g: pd.Series) -> str:
    return f"{g.mean():.3f} ± {g.std(ddof=1):.3f}" if len(g) > 1 else f"{g.mean():.3f}"


def _table(df: pd.DataFrame, names: list[str], cols: list[str]) -> str:
    names = [n for n in names if n in set(df.experiment)]
    lines = ["| experiment | " + " | ".join(c.replace("test_", "").replace("probe_", "probe ") for c in cols)
             + " | params |", "|---|" + "---|" * (len(cols) + 1)]
    for n in names:
        g = df[df.experiment == n]
        cells = [_fmt(g[c].dropna()) if c in g and g[c].notna().any() else "–" for c in cols]
        p = g["n_params"].dropna()
        lines.append(f"| `{n}` | " + " | ".join(cells) + f" | {int(p.iloc[0]):,} |" if len(p) else
                     f"| `{n}` | " + " | ".join(cells) + " | – |")
    return "\n".join(lines)


def paired_table(df: pd.DataFrame, metric: str = "test_auroc") -> str:
    piv = df.pivot(index="seed", columns="experiment", values=metric)
    lines = [f"| A vs B ({metric}) | mean Δ (A−B) | SD of Δ | seeds A>B |", "|---|---|---|---|"]
    for a, b in PAIRS:
        if a in piv and b in piv:
            d = (piv[a] - piv[b]).dropna()
            lines.append(f"| `{a}` vs `{b}` | {d.mean():+.4f} | "
                         f"{d.std(ddof=1) if len(d) > 1 else float('nan'):.4f} | {(d > 0).sum()}/{len(d)} |")
    return "\n".join(lines)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _bars(ax, df, names, metric, ylabel=None, ylim=None):
    """Mean +/- SD marker with per-seed dots (no bars: the y-axis is deliberately not zero-based)."""
    names = [n for n in names if n in set(df.experiment)]
    for i, n in enumerate(names):
        v = df[df.experiment == n][metric].dropna().to_numpy()
        if len(v) == 0:
            continue
        c = _color(n)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=12, facecolors="none",
                   edgecolors=MUTED, linewidths=0.8, zorder=3)
        if len(v) > 1:
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color=c, capsize=4, lw=1.6, zorder=4)
        ax.scatter([i], [v.mean()], s=48, color=c, zorder=5, edgecolors="white", linewidths=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("graph_", "g:") for n in names], rotation=35, ha="right")
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(ylabel or metric, color=MUTED, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax)


def _legend(fig, has_controls: bool, refs: bool = False):
    from matplotlib.lines import Line2D
    dot = lambda c, l: Line2D([], [], marker="o", ls="", color=c, markersize=7, label=l)
    items = [dot(C_BASE, "baseline"), dot(C_GRAPH, "graph fusion (informed edges)")]
    if has_controls:
        items.append(dot(C_CTRL, "graph fusion (control structure)"))
    items.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=MUTED,
                        markersize=5, label="single seed"))
    if refs:
        items += [Line2D([], [], color=MUTED, ls="--", lw=1, label="late_fusion mean"),
                  Line2D([], [], color=MUTED, ls=":", lw=1, label="concat_mlp mean")]
    fig.legend(handles=items, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3 if len(items) > 4 else len(items),
               frameon=False, fontsize=8, labelcolor=MUTED)


def _floor(df, names, metric):
    v = df[df.experiment.isin(names)][metric].dropna()
    lo, hi = v.min(), v.max()
    pad = 0.15 * (hi - lo)
    return max(0.0, lo - pad)


def make_report(results_dir: str | Path) -> None:
    out = Path(results_dir)
    df = pd.read_csv(out / "results.csv")
    n_seeds = df.seed.nunique()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, m, lab in zip(axes, ("test_auroc", "test_auprc"), ("test AUROC", "test AUPRC")):
        _bars(ax, df, MAIN, m, lab, ylim=(_floor(df, MAIN, m), None))
    fig.suptitle(f"Modality ablations and fusion baselines (mean ± SD over {n_seeds} splits; dots = seeds)",
                 fontsize=10, color=INK, y=0.995)
    _legend(fig, False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out / "fig_main_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.1))
    _bars(ax, df, SENS, "test_auroc", "test AUROC", ylim=(_floor(df, SENS, "test_auroc"), None))
    for ref, ls in (("late_fusion", "--"), ("concat_mlp", ":")):
        if ref in set(df.experiment):
            ax.axhline(df[df.experiment == ref].test_auroc.mean(), color=MUTED, lw=1, ls=ls)
    fig.suptitle("Sensitivity to graph construction (test AUROC, mean ± SD over %d splits)" % n_seeds, fontsize=10, color=INK, y=0.995)
    _legend(fig, True, refs=True)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    fig.savefig(out / "fig_graph_sensitivity.png", dpi=160)
    plt.close(fig)

    probe_cols = [("probe_lr_auroc", "linear probe, mortality (AUROC)"),
                  ("probe_knn_auroc", "kNN probe, mortality (AUROC)"),
                  ("probe_los_auroc", "linear probe, LOS ≥ median (AUROC)"),
                  ("probe_saps_r2", "ridge probe, SAPS-I (R²)")]
    names = [n for n in MAIN if n != "gbm_concat"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (m, lab) in zip(axes, probe_cols):
        _bars(ax, df, names, m, lab, ylim=(_floor(df, names, m) if m != "probe_saps_r2" else 0.0, None))
    fig.suptitle("Frozen-embedding probes (fit on train embeddings, scored on test)", fontsize=10, color=INK, y=0.995)
    _legend(fig, False)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out / "fig_embedding_quality.png", dpi=160)
    plt.close(fig)

    md = [f"# Results summary ({out.name})", "",
          f"{len(df)} runs, {n_seeds} seeds (each seed = a new stratified split + new initialisation). "
          "Mean ± sample SD over seeds. These are descriptive repeated-split summaries: the splits "
          "overlap and are not independent, so the SDs are not confidence intervals. Generated by "
          "`python -m tsgf.report`.", "",
          "## Main comparison", "", _table(df, MAIN, METRICS[:3] + METRICS[3:]), "",
          "## Graph-construction sensitivity", "",
          _table(df, SENS, ["test_auroc", "test_auprc", "probe_lr_auroc", "probe_los_auroc"]), "",
          "## Paired differences (same split and seed)", "", "Test AUROC:", "", paired_table(df), "",
          "Test AUPRC:", "", paired_table(df, "test_auprc"), ""]
    if "edges_directed" in df:
        e = df[df.edges_directed.notna()].groupby("experiment")[["edges_directed", "edges_cross"]].mean().round(1)
        md += ["## Edge counts (directed, mean over seeds)", "",
               "| experiment | edges | of which cross-modal (×2) |", "|---|---|---|"]
        md += [f"| `{n}` | {r.edges_directed:.0f} | {r.edges_cross * 2:.0f} |" for n, r in e.iterrows()] + [""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    make_report(ap.parse_args(argv).results)


if __name__ == "__main__":
    main()
