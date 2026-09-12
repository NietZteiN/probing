import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption("--run-heavy", action="store_true", default=False,
                     help="run tests that load a tokenizer or torch (compute node only; the login node caps memory at 8 GB)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-heavy"):
        return
    skip = pytest.mark.skip(reason="needs --run-heavy (tokenizer/torch; run under sbatch)")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: loads a tokenizer or torch; compute node only")
