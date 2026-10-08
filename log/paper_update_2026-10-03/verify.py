from pathlib import Path
from datetime import datetime,timezone
import subprocess
import json
import sys
root=Path(__file__).resolve().parents[2].parent
report={'started_utc':datetime.now(timezone.utc).isoformat(),'stages':[],'complete':False}
for repo,generator in [('codecue','51_numbers.py'),('probing','51_tables.py')]:
 for name,argv in [('numbers',[sys.executable,'scripts/'+generator]),('selfcheck',[sys.executable,'scripts/99_selfcheck.py']),('build',['make','paper-submission'])]:
  path=root/repo/'log/paper_update_2026-10-03'/f'{name}.out'
  with path.open('w') as stream:
   run=subprocess.run(argv,cwd=root/repo,stdout=stream,stderr=subprocess.STDOUT)
  report['stages'].append({'repo':repo,'name':name,'argv':argv,'exit_code':run.returncode,'log':str(path)})
  (root/'probing/log/paper_update_2026-10-03/verification.json').write_text(json.dumps(report,indent=2)+'\n')
  print(repo,name,'exit',run.returncode,flush=True)
  if run.returncode:sys.exit(run.returncode)
report['complete']=True
(root/'probing/log/paper_update_2026-10-03/verification.json').write_text(json.dumps(report,indent=2)+'\n')
