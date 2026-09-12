#!/usr/bin/env python
"""GPU stage 2: neutral-only probes for one (model, level, regime), every role.

    python scripts/30_train_probes.py --model llama32-3b --level 3 --regime cot [--train train_neutral]
        [--roles v1 v2] [--optimizer sgd|lbfgs] [--epochs 10000] [--positions ...] [--layers ...]

--train train_letter with test group `letter` reproduces Kudo et al. (their probes are trained
and tested on letters). Output: $PROBE_OUT/probes/<model>/L<level>/<regime>/<train>/<role>.json (+.npz)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cueconf.config import OUT_DIR  # noqa: E402
from cueconf.generator import LEVELS  # noqa: E402
from cueconf.probes import train_and_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--regime", default="cot")
    ap.add_argument("--train", default="train_neutral")
    ap.add_argument("--roles", nargs="*", default=None)
    ap.add_argument("--optimizer", default="sgd", choices=["sgd", "lbfgs"])
    ap.add_argument("--epochs", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--positions", nargs="*", default=None)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="indices into the cached layer list")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--test-groups", nargs="*", default=None)
    ap.add_argument("--suffix", default="", help="'__alltok' to probe the E27 all-token caches; output goes to <train><suffix>/")
    a = ap.parse_args()
    base = OUT_DIR / "runs" / a.model / f"L{a.level}" / a.regime
    train_dir = base / (a.train + a.suffix)
    if not (train_dir / "meta.json").exists():
        print(f"FATAL: no cache at {train_dir}; run 20_run_model.py first", file=sys.stderr)
        return 2
    test_dirs = {p.name.removesuffix(a.suffix): p for p in sorted(base.iterdir())
                 if (p / "meta.json").exists() and not p.name.startswith("train_") and p.name.endswith(a.suffix)
                 and (a.suffix or "__" not in p.name)}
    if a.test_groups:
        test_dirs = {g: test_dirs[g] for g in a.test_groups}
    roles = a.roles or [lhs for lhs, _ in LEVELS[a.level]["eqs"]]
    for role in roles:
        out = OUT_DIR / "probes" / a.model / f"L{a.level}" / a.regime / (a.train + a.suffix) / f"{role}.json"
        train_and_eval(train_dir, test_dirs, role, out, seeds=tuple(a.seeds), optimizer=a.optimizer, epochs=a.epochs,
                       lr=a.lr, layers=a.layers, positions=a.positions, control=not a.no_control, standardize=a.standardize)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
