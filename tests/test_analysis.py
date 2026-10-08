"""Regression checks for the paired written-value analysis and its seed pooling."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_written_value_uses_same_pairs_for_estimate_and_interval(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "62_value_written.py"
    spec = importlib.util.spec_from_file_location("written_value", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, output, results = (tmp_path / name for name in ("data", "out", "results"))
    (data / "L3").mkdir(parents=True)
    (data / "L3/test_sets.jsonl").touch()
    (results / "summary").mkdir(parents=True)
    instances = []
    for i in range(80):
        for condition, name in (("inc", "two"), ("neu", "pen")):
            instances.append(SimpleNamespace(id=f"{condition}-{i}", set_id=f"set-{i}",
                                             names={"v2": name}, values={"v2": 5}))
    monkeypatch.setattr(module, "DATA_DIR", data)
    monkeypatch.setattr(module, "OUT_DIR", output)
    monkeypatch.setattr(module, "RESULTS_DIR", results)
    monkeypatch.setattr(module, "read_jsonl", lambda path: instances)
    for regime in ("cot", "cot_s11", "cot_s13"):
        run = output / "runs/olmo2-1b-it/L3" / regime
        for group, condition in (("incongruent@v2", "inc"), ("neutral", "neu")):
            dest = run / group
            dest.mkdir(parents=True)
            rows = []
            for i in range(80):
                rows.append({"id": f"{condition}-{i}", "lure": 2 if condition == "inc" else None,
                             "pred_is_lure": condition == "inc" and i % 4 < 2, "pred": 0,
                             "generation": "two=5" if condition == "inc" else
                                           ("pen=5" if i % 4 < 3 else "pen=0")})
            (dest / "behavior.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        archive = run / "incongruent@v2__smoke64"
        archive.mkdir()
        (archive / "behavior.jsonl").write_text(json.dumps(rows[0]))
    assert module.main() == 0
    rows = json.loads((results / "summary/value_written.json").read_text())
    for row in rows:
        assert row["raw_lure_wrote"] == 0.5
        assert row["n_matched_wrote"] == 60
        assert row["excess_wrote"] == pytest.approx(2 / 3)
        lo, hi = row["excess_wrote_ci"]
        assert lo < row["excess_wrote"] < hi
    pooled, = json.loads((results / "summary/value_written_pooled.json").read_text())
    assert pooled["seeds"] == [7, 11, 13]
    assert pooled["n_matched_wrote"] == 180
    assert pooled["n_wrote"] == 240
    assert pooled["excess_wrote"] == pytest.approx(2 / 3)
    # Identical seed replicates must move together: pooling them does not shrink the CI.
    assert pooled["excess_wrote_ci"] == rows[0]["excess_wrote_ci"]
