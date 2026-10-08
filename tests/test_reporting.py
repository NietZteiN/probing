"""The preregistered accuracy exclusion must be checked in the actual paper folder."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("accuracy,in_table,should_fail", [
    (.47, True, True), (.47, False, False), (.90, True, False),
])
def test_main_table_accuracy_floor(tmp_path, monkeypatch, accuracy, in_table, should_fail):
    script = Path(__file__).resolve().parents[1] / "scripts/99_selfcheck.py"
    spec = importlib.util.spec_from_file_location("reporting_selfcheck", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "scratch")
    monkeypatch.setattr(module, "load_config", lambda _: {"models": {"low": {}, "high": {}}})
    summary = tmp_path / "results/summary"
    summary.mkdir(parents=True)
    sweep = {"low/cot": {"seeds": [7, 11, 13], "contrasts": {},
                         "groups": {"neutral": {"acc_mean": accuracy}}},
             "high/cot": {"seeds": [7, 11, 13], "contrasts": {},
                          "groups": {"neutral": {"acc_mean": .98}}}}
    (summary / "seed_sweep_L3.json").write_text(json.dumps(sweep))
    tables = tmp_path / "paper/tables"
    tables.mkdir(parents=True)
    (tables / "behavior.tex").write_text("low & CoT \\\\\n" if in_table else "high & CoT \\\\\n")
    module.check_sweeps(3)
    assert bool(module.FAIL) is should_fail
    if should_fail:
        assert "90% CoT accuracy floor" in module.FAIL[0]
