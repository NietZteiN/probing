"""Paths and config loading. Every root comes from scripts/env.sh (PROBE_* variables); the
defaults here only make the package importable without sourcing it (tests, login-node checks)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(os.environ.get("PROBE_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
# Large, disposable outputs (hidden states, probe weights, patching sweeps) live on scratch.
OUT_DIR = Path(os.environ.get("PROBE_OUT", f"/scratch/juno/{os.environ.get('USER', 'x')}/probing"))
RESULTS_DIR = PROJECT_ROOT / "results"   # small JSON/CSV summaries; committed selectively


def load_config(name: str) -> dict:
    with (CONFIG_DIR / name).open() as f:
        return yaml.safe_load(f)


def model_entry(key: str) -> dict:
    models = load_config("models.yaml")["models"]
    if key not in models:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(models)}")
    return {"key": key, **models[key]}
