from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[2]
for repo in ('codecue','probing'):
 path=root.parent/repo/'log/round5_2026-10-03/final_build.out'
 with path.open('w') as f:
  run=subprocess.run(['make','paper-submission'],cwd=root.parent/repo,stdout=f,stderr=subprocess.STDOUT)
 if run.returncode:raise RuntimeError(f'{repo} strict build failed; see {path}')
 print(repo,'strict build passed',flush=True)
