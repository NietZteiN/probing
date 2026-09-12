"""scripts/env.sh must export only PROBE_-prefixed roots (plus the standard cache variables)."""
import re
from pathlib import Path

ALLOWED = {"PATH", "PYTHONPATH", "HF_HOME", "HF_HUB_OFFLINE", "TMPDIR", "TORCHINDUCTOR_CACHE_DIR",
           "TRITON_CACHE_DIR", "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED", "PYTHONUNBUFFERED"}


def test_env_exports_are_prefixed():
    raw = (Path(__file__).resolve().parents[1] / "scripts" / "env.sh").read_text()
    src = "\n".join(l for l in raw.splitlines() if not l.lstrip().startswith("#"))
    names = re.findall(r"^export\s+([A-Z_]+)=", src, flags=re.M)
    bad = [n for n in names if not n.startswith("PROBE_") and n not in ALLOWED]
    assert not bad, bad
    assert "SCRATCH=" not in src.replace("PROBE_SCRATCH=", "")
