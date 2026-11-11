import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_results", ROOT / "scripts/verify_results.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_reference_runs_match_published_tables():
    verifier.verify(ROOT)


@pytest.mark.parametrize("mutation", ["missing", "readme"])
def test_reference_drift_is_detected(tmp_path, mutation):
    for mode in ("quick", "full"):
        run = tmp_path / "results" / mode
        run.mkdir(parents=True)
        for name in ("config_used.yaml", "results.csv"):
            shutil.copy2(ROOT / "results" / mode / name, run / name)
    shutil.copy2(ROOT / "README.md", tmp_path / "README.md")
    if mutation == "missing":
        csv = tmp_path / "results/quick/results.csv"
        lines = csv.read_text(encoding="utf-8").splitlines()
        csv.write_text("\n".join([lines[0], *lines[2:]]) + "\n", encoding="utf-8")
    else:
        readme = tmp_path / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8").replace("0.664 ± 0.022", "0.964 ± 0.022"), encoding="utf-8")
    with pytest.raises(ValueError):
        verifier.verify(tmp_path)
