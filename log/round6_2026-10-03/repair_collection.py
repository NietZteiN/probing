from pathlib import Path
import subprocess,sys,os
root=Path(__file__).resolve().parents[2].parent
log=root/'probing/log/round6_2026-10-03'
with (log/'collection_regression_tests.out').open('w') as stream:
 r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_round6.py'],cwd=root/'codecue',stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':str(root/'codecue/src')})
if r.returncode:sys.exit(r.returncode)
with (log/'collection_retry.out').open('w') as stream:
 r=subprocess.run([sys.executable,'scripts/81_round6_completion.py'],cwd=root/'probing',stdout=stream,stderr=subprocess.STDOUT)
sys.exit(r.returncode)
