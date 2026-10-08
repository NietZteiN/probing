from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[2]
log=root/'log/round6_2026-10-03'
with (log/'precision_tests.out').open('w') as stream:
 run=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_cache_precision.py','--run-heavy'],cwd=root,stdout=stream,stderr=subprocess.STDOUT)
if run.returncode:sys.exit(run.returncode)
(log/'precision_ready.ok').write_text('Both float32 preservation and nonfinite probe guard tests passed.\n')
