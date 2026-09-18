#!/usr/bin/env python
"""E31 (Amendment 4): the value-only chain on the eight-model panel, as two dependency chains so
that at most two GPU jobs run at once (the project's GPU budget).

    python scripts/slurm/submit_e31.py [--dry-run]

Chain A (a30 / h100, the Llama family): caches for the two base models (seed 7, with the
probe-train groups), behaviour only for the other seeds and the instruct models, then probes on
the two cached models and patching on the 3B. Chain B (h200, one job at a time): caches for
OLMo-2-1B (the arithmetic exception), behaviour for the other three, then its probes.
Reason for h200 in this manifest: those four models are assigned to h200 in configs/models.yaml
and have only ever run there; one job at a time keeps the account inside the shared pool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "slurm"))
from cueconf.config import load_config  # noqa: E402
from submit import submit  # noqa: E402

PY = "python"
RUN = [PY, "scripts/20_run_model.py", "--level", "3"]
SEEDS = ["simple_s11", "simple_s13"]


def run_cached(k, part, dep, dry):
    return submit(f"e31run_{k}", part, RUN + ["--model", k, "--regimes", "simple"], dependency=dep, dry_run=dry)


def run_behaviour(k, part, dep, dry, regimes):
    return submit(f"e31beh_{k}", part, RUN + ["--model", k, "--regimes", *regimes, "--no-train", "--no-forced"],
                  dependency=dep, dry_run=dry)


def probes(k, part, dep, dry):
    return submit(f"e31probes_{k}", part, [PY, "scripts/30_train_probes.py", "--model", k, "--level", "3",
                                            "--regime", "simple", "--train", "train_neutral"], dependency=dep, dry_run=dry)


def patch(k, part, dep, dry):
    return submit(f"e31patch_{k}", part, [PY, "scripts/40_patch.py", "--model", k, "--level", "3", "--regime", "simple"],
                  dependency=dep, dry_run=dry)


def after(*ids):
    """afterany on the previous link of the chain, plus afterok on any cache job a stage reads."""
    parts = [f"{kind}:{i}" for kind, i in ids if i]
    return ",".join(parts) or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run
    part = {k: v["partition"] for k, v in load_config("models.yaml")["models"].items()}

    # ---- chain A: Llama family on a30 / h100
    a1 = run_cached("llama32-3b", part["llama32-3b"], None, dry)
    a2 = run_behaviour("llama32-3b", part["llama32-3b"], after(("afterany", a1)), dry, SEEDS)
    a3 = run_behaviour("llama32-3b-it", part["llama32-3b-it"], after(("afterany", a2)), dry, ["simple", *SEEDS])
    a4 = run_cached("llama31-8b", part["llama31-8b"], after(("afterany", a3)), dry)
    a5 = run_behaviour("llama31-8b", part["llama31-8b"], after(("afterany", a4)), dry, SEEDS)
    a6 = run_behaviour("llama31-8b-it", part["llama31-8b-it"], after(("afterany", a5)), dry, ["simple", *SEEDS])
    a7 = probes("llama32-3b", part["llama32-3b"], after(("afterok", a1), ("afterany", a6)), dry)
    a8 = probes("llama31-8b", part["llama31-8b"], after(("afterok", a4), ("afterany", a7)), dry)
    patch("llama32-3b", part["llama32-3b"], after(("afterok", a1), ("afterany", a8)), dry)

    # ---- chain B: the h200 models, one at a time
    b1 = run_cached("olmo2-1b-it", part["olmo2-1b-it"], None, dry)
    b2 = run_behaviour("olmo2-1b-it", part["olmo2-1b-it"], after(("afterany", b1)), dry, SEEDS)
    b3 = run_behaviour("olmo2-7b-it", part["olmo2-7b-it"], after(("afterany", b2)), dry, ["simple", *SEEDS])
    b4 = run_behaviour("gemma3-4b-it", part["gemma3-4b-it"], after(("afterany", b3)), dry, ["simple", *SEEDS])
    b5 = run_behaviour("gemma3-12b-it", part["gemma3-12b-it"], after(("afterany", b4)), dry, ["simple", *SEEDS])
    probes("olmo2-1b-it", part["olmo2-1b-it"], after(("afterok", b1), ("afterany", b5)), dry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
