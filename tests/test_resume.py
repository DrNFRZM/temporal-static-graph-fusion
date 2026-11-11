"""Protect experiment records when extending or accidentally reusing a run."""
import copy

import pandas as pd
import pytest
import yaml

from tsgf.run import main, resume_config
from tsgf.utils import load_config


@pytest.fixture
def recorded(tmp_path):
    cfg = {"name": "test", "output_dir": str(tmp_path), "seeds": [0],
           "data": {"sets": ["a"]}, "train": {"epochs": 3},
           "experiments": [{"name": "graph", "model": "graph", "graph": {"kind": "random"}}]}
    (tmp_path / "config_used.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    pd.DataFrame([{"experiment": "graph", "seed": 0, "test_auroc": 0.75}]).to_csv(
        tmp_path / "results.csv", index=False)
    return cfg


@pytest.mark.parametrize("section,key,value", [("train", "epochs", 8), ("data", "sets", ["b"])])
def test_changed_protocol_is_rejected_without_touching_evidence(tmp_path, recorded, section, key, value):
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    cfg = copy.deepcopy(recorded)
    cfg[section][key] = value
    with pytest.raises(ValueError, match="new --output-dir"):
        resume_config(tmp_path, cfg, cfg["experiments"])
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_extension_retains_previous_seeds_and_experiment_definitions(tmp_path, recorded):
    cfg = copy.deepcopy(recorded)
    cfg["seeds"] = [1]
    extra = {"name": "flat", "model": "concat_mlp"}
    merged = resume_config(tmp_path, cfg, [extra])
    assert merged["seeds"] == [0, 1]
    assert merged["experiments"] == recorded["experiments"] + [extra]


def test_renamed_method_cannot_replace_existing_method(tmp_path, recorded):
    changed = [{"name": "graph", "model": "graph", "graph": {"kind": "none"}}]
    with pytest.raises(ValueError, match="experiment 'graph' changed"):
        resume_config(tmp_path, recorded, changed)


def test_unattributed_results_are_rejected(tmp_path):
    (tmp_path / "results.csv").write_text("experiment,seed\ngraph,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no config_used.yaml"):
        resume_config(tmp_path, {}, [])


def test_duplicate_results_are_rejected(tmp_path, recorded):
    (tmp_path / "results.csv").write_text("experiment,seed\ngraph,0\ngraph,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        resume_config(tmp_path, recorded, recorded["experiments"])


def test_cli_typo_fails_before_output_is_created(tmp_path):
    out = tmp_path / "new"
    with pytest.raises(SystemExit):
        main(["--config", "configs/quick.yaml", "--only", "misspelled", "--output-dir", str(out)])
    assert not out.exists()


def test_cli_preserves_completed_run_and_rejects_changed_epochs(tmp_path, monkeypatch, synthetic_raw):
    cfg = load_config("configs/quick.yaml")
    cfg["seeds"] = [0]
    cfg["model"].update(d_node=8, d_hidden=16, gnn_layers=1, gnn_heads=2)
    cfg["train"].update(epochs=1, threads=1)
    cfg["experiments"] = [{"name": "static", "model": "static_mlp"}]
    config = tmp_path / "input.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "run"
    monkeypatch.setattr("tsgf.run.load_raw", lambda *_: synthetic_raw)
    args = ["--config", str(config), "--output-dir", str(out)]
    main(args)
    evidence = {name: (out / name).read_bytes() for name in ("results.csv", "environment.json", "config_used.yaml")}
    def no_training(*_args, **_kwargs):
        pytest.fail("a completed experiment must not be trained again")
    monkeypatch.setattr("tsgf.run.run_one", no_training)
    main(args)
    assert evidence == {name: (out / name).read_bytes() for name in evidence}
    all_files = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match="new --output-dir"):
        main([*args, "--epochs", "2"])
    assert all_files == {p.name: p.read_bytes() for p in out.iterdir()}
