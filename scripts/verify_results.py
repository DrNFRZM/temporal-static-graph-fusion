"""Check saved runs and README AUROC/AUPRC arithmetic without data or training.

This checks internal consistency, not the original training execution or clinical validity.
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("static_mlp", "temporal_only", "concat_mlp", "late_fusion", "graph_corr_k4", "gbm_concat")


def verify(root=ROOT):
    readme = (root / "README.md").read_text(encoding="utf-8")
    for mode in ("quick", "full"):
        run = root / "results" / mode
        cfg = yaml.safe_load((run / "config_used.yaml").read_text(encoding="utf-8"))
        with (run / "results.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {(e["name"], s) for e in cfg["experiments"] for s in cfg["seeds"]}
        observed = [(r["experiment"], int(r["seed"])) for r in rows]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError(f"{mode}: duplicate or missing experiment/seed rows")
        for row in rows:
            for key in ("test_auroc", "test_auprc", "val_auroc"):
                value = float(row[key])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{mode}: invalid {key}")
        section = readme.split(f"### {mode.title()} mode", 1)[1].split("\n### ", 1)[0]
        for model in MODELS:
            cells = []
            for metric in ("test_auroc", "test_auprc"):
                values = [float(r[metric]) for r in rows if r["experiment"] == model]
                cells.append(f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}")
            line = next((line.replace("**", "") for line in section.splitlines()
                         if line.replace("**", "").startswith(f"| `{model}` ")), "")
            if [c.strip() for c in line.split("|")[2:4]] != cells:
                raise ValueError(f"{mode}: README AUROC/AUPRC differs for {model}")
        print(f"{mode}: verified {len(rows)} result rows and six README AUROC/AUPRC rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
